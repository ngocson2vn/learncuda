#include <cstdio>
#include "cute/layout.hpp"

using namespace cute;

FILE* get_file_ptr() {
  FILE *fptr;
  const char* tex_file = "layout.tex";
  fptr = fopen(tex_file, "w");
  if(fptr == NULL) {
    printf("Failed to open %s\n", tex_file);
    exit(1);
  }

  return fptr;
}

namespace util {

// Generic 2D Layout to LaTeX printer
template <class LayoutA, class TikzColorFn = TikzColor_BWx8>
CUTE_HOST_DEVICE
void
fprint_latex(LayoutA const& layout_a,   // (m,n) -> idx
            TikzColorFn color = {})    // lambda(idx) -> tikz color string
{
  FILE* os = get_file_ptr();
  CUTE_STATIC_ASSERT_V(rank(layout_a) <= Int<2>{});
  auto layout = append<2>(layout_a, Layout<_1,_0>{});

  // Commented print(layout)
  // fprintf(os, "%% Layout: %s\n", layout);
  // Header
  fprintf(os, "\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  // Layout
  for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
      int idx = layout(i,j);
      fprintf(os, "\\node[fill=%s] at (%d,%d) {%d};\n",
             color(idx), i, j, idx);
    }
  }
  // Grid
  fprintf(os, "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n",
         int(size<0>(layout)), int(size<1>(layout)));
  // Labels
  for (int i =  0, j = -1; i < size<0>(layout); ++i) {
    fprintf(os, "\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int i = -1, j =  0; j < size<1>(layout); ++j) {
    fprintf(os, "\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }

  // Footer
  fprintf(os, "\\end{tikzpicture}\n"
         "\\end{document}\n");
}

} // end namespace util