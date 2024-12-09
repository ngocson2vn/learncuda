#include "cute/layout.hpp"

using namespace cute;

namespace util {

// Generic 2D Layout to LaTeX printer
template <class LayoutA, class TikzColorFn = TikzColor_BWx8>
CUTE_HOST_DEVICE
void
print_latex(LayoutA const& layout_a,   // (m,n) -> idx
            TikzColorFn color = {})    // lambda(idx) -> tikz color string
{
  CUTE_STATIC_ASSERT_V(rank(layout_a) <= Int<2>{});
  auto layout = append<2>(layout_a, Layout<_1,_0>{});

  // Commented print(layout)
  printf("%% Layout: "); print(layout); printf("\n");
  // Header
  printf("\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  // Layout
  for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
      int idx = layout(i,j);
      printf("\\node[fill=%s] at (%d,%d) {%d};\n",
             color(idx), i, j, idx);
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n",
         int(size<0>(layout)), int(size<1>(layout)));
  // Labels
  for (int i =  0, j = -1; i < size<0>(layout); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int i = -1, j =  0; j < size<1>(layout); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }

  // Footer
  printf("\\end{tikzpicture}\n"
         "\\end{document}\n");
}

} // end namespace util