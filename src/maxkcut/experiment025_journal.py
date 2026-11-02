"""Crash-resilient, original-graph-verified best-so-far progress journal.

Every meaningful event (initializer candidate, flip sweep, KL pass, ILS cycle)
gets a line in events.jsonl. The best feasible *binary partition* is saved in
an immutable versioned NPZ each time a strictly better original signed cut is
verified. No last-iterate assumptions or unverified auxiliary objective scores.

The event stream is append-only; on recovery, the best pointer is verified
against the original graph, factor/seed protocol, and partition SHA256.
A run's full stage partition archive is saved separately by Experiment025.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np

from .experiment023_methods import _score, sha, atomic_json


class IterationJournal:
    def __init__(self, path, edges, graph_size, fingerprint):
        self.path=Path(path)
        self.path.mkdir(parents=True,exist_ok=True)
        self.edges=edges
        self.n=int(graph_size)
        self.fingerprint=str(fingerprint)
        self.best_cut=float("-inf")
        self.best_partition=None
        self.best_event=None
        self.events=0
        self.improvements=0
        marker=self.path/"identity.json"
        if marker.is_file():
            expected=json.loads(marker.read_text(encoding="utf8"))
            if expected!=dict(fingerprint=self.fingerprint,graph_size=self.n):
                raise ValueError("Journal graph or protocol identity changed")
        else:
            atomic_json(marker,dict(fingerprint=self.fingerprint,graph_size=self.n))
        eventfile=self.path/"events.jsonl"
        if eventfile.is_file():
            with eventfile.open(encoding="utf8") as fh:
                for line in fh:
                    if line.strip():
                        json.loads(line)  # fail if the last event is truncated
                        self.events+=1
        pointer=self.path/"best.json"
        if pointer.is_file():
            meta=json.loads(pointer.read_text(encoding="utf8"))
            if meta["fingerprint"]!=self.fingerprint:
                raise ValueError("Invalid journal best fingerprint")
            file=self.path/meta["partition"]
            if not file.is_file() or sha(file)!=meta["sha256"]:
                raise ValueError("Missing/corrupt archived best partition")
            with np.load(file,allow_pickle=False) as src:
                label=src["labels"].astype(np.int8)
            score=self._verify(label)
            if not np.isclose(score,meta["best_cut"],rtol=0,atol=1e-8):
                raise ValueError("Archived best-cut metadata disagree with original graph")
            self.best_partition=label.copy()
            self.best_cut=float(score)
            self.best_event=meta["event"]
            self.improvements=int(meta["improvements"])

    def _verify(self, partition):
        x=np.asarray(partition,dtype=np.int8)
        if x.shape!=(self.n,) or not np.isin(x,[0,1]).all():
            raise ValueError("Invalid binary cut partition in progress observer")
        return float(_score(self.edges,x))

    def observe(self, event, partition):
        """Append every iteration; persist partition only on new best."""
        record=dict(event)
        stage=str(record.get("stage") or "unknown")
        labels=np.asarray(partition,dtype=np.int8)
        cut=self._verify(labels)
        current_reported=record.get("current_cut")
        # Some heuristics report the current/WALK score while emitting their
        # separate incumbent best partition. Both are retained, not confused.
        record["reported_current_cut"]=current_reported
        record["observed_partition_cut"]=cut
        record["stage"]=stage
        record["fingerprint"]=self.fingerprint
        improved=cut>self.best_cut+1e-8
        if improved:
            version=self.improvements+1
            partition_name=f"best_{version:06d}.npz"
            output=self.path/partition_name
            tmp=self.path/(partition_name+".temporary.npz")
            np.savez_compressed(tmp,labels=labels.astype(np.uint8))
            os.replace(tmp,output)
            self.best_cut=cut
            self.best_partition=labels.copy()
            self.best_event=dict(stage=stage,iteration=record.get("iteration"),
                pass_index=record.get("pass"),cycle=record.get("cycle"),
                sweep=record.get("sweep"),seed=record.get("seed"))
            self.improvements=version
            atomic_json(self.path/"best.json",dict(
                fingerprint=self.fingerprint,best_cut=cut,
                partition=partition_name,sha256=sha(output),
                event=self.best_event,improvements=version))
        record["improved_best"]=bool(improved)
        record["best_so_far_cut"]=self.best_cut
        record["event_number"]=self.events+1
        # Retain the event as an audit trail even if the cut was not improved.
        with (self.path/"events.jsonl").open("a",encoding="utf8") as stream:
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events+=1
        return self.best_cut

    def assert_best(self):
        if self.best_partition is None or self.best_cut==float("-inf"):
            raise RuntimeError("No verified best-so-far solution")
        if abs(self._verify(self.best_partition)-self.best_cut)>1e-8:
            raise AssertionError("Best-so-far partition lost its cut value")
        return self.best_partition.copy(),float(self.best_cut)

    def summary(self):
        best,score=self.assert_best()
        return dict(best_seen_cut=score,best_seen_event=self.best_event,
                    iteration_events=self.events,
                    best_partition_improvements=self.improvements,
                    progress_log=str(self.path/"events.jsonl"),
                    best_meta=str(self.path/"best.json"))


def refine_with_journal(graph, edges, initial, method, restart, graph_number,
                        cfg, journal):
    """Use the actual common Flip/KL/ILS algorithms and persist each cycle.

    The callback observes a verified score at every meaningful outer
    iteration, and archives the highest-scoring binary partition. The
    candidate ILS walk may deteriorate; only the maximum ever found wins.
    """
    from .local_search import improve_by_flips
    from .eva_elite_search import optimize_kl_fast, optimize_elite
    x=np.asarray(initial,dtype=np.int8).copy()
    start_cut=journal._verify(x)
    journal.observe(dict(stage="Round",iteration=0,
                         current_cut=start_cut),x.copy())
    def random(tag):
        return np.random.default_rng(np.random.SeedSequence(
            [2026,graph_number,restart,tag]))
    flip, flip_d=improve_by_flips(
        graph,x.copy(),random(1313),max_sweeps=20,
        on_iteration=journal.observe)
    journal.observe(dict(stage="FlipBest",iteration=flip_d["sweeps"],
                         current_cut=journal._verify(flip)),flip.copy())
    kl, kl_d=optimize_kl_fast(
        graph,flip.copy(),random(101),passes=cfg["kl_passes"],
        on_iteration=journal.observe)
    if journal._verify(kl)<journal._verify(flip)-1e-8:
        kl=flip.copy()
    journal.observe(dict(stage="KLBest",
                         iteration=kl_d["passes_improved"],
                         current_cut=journal._verify(kl)),kl.copy())
    refiner="Tabu-KL" if method=="control-tabu" else "ILS-KL"
    final, search=optimize_elite(
        graph,kl.copy(),[(kl,journal._verify(kl))],random(1401),
        method=refiner,
        cycles=cfg["ils_cycles"],kl_passes=cfg["ils_kl_passes"],
        kick_sizes=(8,16,32,64),temperature=12.,
        **(dict(tabu_steps=1000,tabu_stagnation=250)
           if refiner=="Tabu-KL" else {}),
        on_iteration=journal.observe)
    journal.observe(dict(stage="ILSBest",iteration=search["cycles_completed"],
                         current_cut=journal._verify(final)),final.copy())
    best, best_score=journal.assert_best()
    if best_score<journal._verify(final)-1e-8:
        raise AssertionError("Journal lost a feasible ILS improvement")
    # Existing archive verifier requires exactly Round, Flip, KL, Final.
    # Here 'Final' means globally BEST SEEN, not the final walking state.
    stages=dict(Round=start_cut,Flip=journal._verify(flip),
                KL=journal._verify(kl),Final=best_score)
    partitions=dict(Round=x,Flip=flip,KL=kl,Final=best)
    if not (stages["Round"]<=stages["Flip"]+1e-8 and
            stages["Flip"]<=stages["KL"]+1e-8 and
            stages["KL"]<=stages["Final"]+1e-8):
        raise AssertionError("Best-so-far stage ledger is non-monotone")
    diagnostics=dict(refiner=refiner,flip=flip_d,kl=kl_d,
                     ils_cycles_completed=search["cycles_completed"],
                     ils_trace=search["trace"],
                     ils_improvement_events=search["best_events"],
                     **journal.summary())
    return best,stages,diagnostics,partitions
