ROOT=main.tex

.PHONY: verify pdf clean
verify:
	python verify_results.py
pdf: verify
	latexmk -pdf -file-line-error -halt-on-error -interaction=nonstopmode $(ROOT)
clean:
	latexmk -c $(ROOT)
