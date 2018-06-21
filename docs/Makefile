# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Pibronic
SOURCEDIR     = .
BUILDDIR      = _build
GHPAGES_BUILDDIR = ../../Pibronic-docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

gh-pages:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(GHPAGES_BUILDDIR)" $(SPHINXOPTS) $(O)
	cd "$(GHPAGES_BUILDDIR)/html" && git add . && git commit -m "Updated gh-pages" && git push

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)