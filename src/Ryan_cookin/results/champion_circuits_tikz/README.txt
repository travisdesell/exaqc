Champion circuits as Quantikz/LaTeX

Each .tex file is a `standalone` document with a single quantikz
diagram. Encoder gates (enc_*) render symbolically, e.g.

  enc_ry on q0 reading x[0]
    -> R_y(a_n x_0 + b_n)

  enc_rot
    -> Rot(a^alpha_n x_f + b^alpha_n,
           a^beta_n  x_f + b^beta_n,
           a^gamma_n x_f + b^gamma_n)

  enc_xx
    -> R_{XX}(a^0_n x_{f_0} + a^1_n x_{f_1} + b_n)

so the data dependence is explicit rather than evaluated to a number
at some placeholder x (which is what the matplotlib PNGs in
champion_circuits/ show).

Trained-scalar ansatz gates (ry, rx, rz, rxx, ryy, rzz) DO show their
converged numeric value, since for those it's not a function of x.

Stage A renders the hand-built RY-CNOT chain with R_y(phi) placeholders.
Stages B and C render only the genome (ansatz) part; the external
encoder is summarised in the header comment because expanding its
gates inline would blow the diagram up to dozens of columns.

Compile with:
    pdflatex -output-directory <stage_dir> <stage_dir>/<dataset>.tex

You'll need the `quantikz` LaTeX package installed (it ships with
recent TeX Live distributions; otherwise `tlmgr install quantikz`).
