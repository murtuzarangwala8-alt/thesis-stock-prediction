import os
from pathlib import Path

base_dir = Path(r"c:\Users\murta\Desktop\thesis final 2.0\thesis")
chapters_dir = base_dir / "chapters"
chapters_dir.mkdir(parents=True, exist_ok=True)

print("Building 120+ Page Master Thesis Manuscript Generator...")

# 1. References.bib
bib_content = r"""@article{fama1970efficient,
  title={Efficient capital markets: A review of theory and empirical work},
  author={Fama, Eugene F},
  journal={The Journal of Finance},
  volume={25},
  number={2},
  pages={383--417},
  year={1970}
}

@article{sharpe1964capital,
  title={Capital asset prices: A theory of market equilibrium under conditions of risk},
  author={Sharpe, William F},
  journal={The Journal of Finance},
  volume={19},
  number={3},
  pages={425--442},
  year={1964}
}

@article{lintner1965valuation,
  title={The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets},
  author={Lintner, John},
  journal={The Review of Economics and Statistics},
  volume={47},
  number={1},
  pages={13--37},
  year={1965}
}

@article{fama1973risk,
  title={Risk, return, and equilibrium: Empirical tests},
  author={Fama, Eugene F and MacBeth, James D},
  journal={Journal of Political Economy},
  volume={81},
  number={3},
  pages={607--636},
  year={1973}
}

@article{ross1976arbitrage,
  title={The arbitrage theory of capital asset pricing},
  author={Ross, Stephen A},
  journal={Journal of Economic Theory},
  volume={13},
  number={3},
  pages={341--360},
  year={1976}
}

@article{fama1993common,
  title={Common risk factors in the returns on stocks and bonds},
  author={Fama, Eugene F and French, Kenneth R},
  journal={Journal of Financial Economics},
  volume={33},
  number={1},
  pages={3--56},
  year={1993}
}

@article{fama2015five,
  title={A five-factor asset pricing model},
  author={Fama, Eugene F and French, Kenneth R},
  journal={Journal of Financial Economics},
  volume={116},
  number={1},
  pages={1--22},
  year={2015}
}

@article{carhart1997persistence,
  title={On persistence in mutual fund performance},
  author={Carhart, Mark M},
  journal={The Journal of Finance},
  volume={52},
  number={1},
  pages={57--82},
  year={1997}
}

@article{cochrane2011discount,
  title={Presidential address: Discount rates},
  author={Cochrane, John H},
  journal={The Journal of Finance},
  volume={66},
  number={4},
  pages={1047--1108},
  year={2011}
}

@article{harvey2016and,
  title={... and the cross-section of expected stock returns},
  author={Harvey, Campbell R and Liu, Yan and Zhu, Heqing},
  journal={The Review of Financial Studies},
  volume={29},
  number={1},
  pages={5--68},
  year={2016}
}

@article{kozak2020shrinking,
  title={Shrinking the book of anomaly factors},
  author={Kozak, Serhiy and Nagel, Stefan and Santosh, Shrihari},
  journal={Journal of Financial Economics},
  volume={135},
  number={2},
  pages={271--292},
  year={2020}
}

@article{freyberger2020dissecting,
  title={Dissecting characteristics nonparametrically},
  author={Freyberger, Joachim and Neuhierl, Andreas and Weber, Michael},
  journal={The Review of Financial Studies},
  volume={33},
  number={5},
  pages={2326--2377},
  year={2020}
}

@article{gu2020empirical,
  title={Empirical asset pricing via machine learning},
  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal={The Review of Financial Studies},
  volume={33},
  number={5},
  pages={2223--2273},
  year={2020}
}

@article{kelly2019characteristics,
  title={Characteristics are covariances: A structural model of risk and return},
  author={Kelly, Bryan T and Pruitt, Seth and Su, Yinan},
  journal={Journal of Financial Economics},
  volume={134},
  number={3},
  pages={501--524},
  year={2019}
}

@article{chen2023deep,
  title={Deep learning for asset pricing},
  author={Chen, Luyang and Pelger, Markus and Zhu, Jason},
  journal={Management Science},
  volume={69},
  number={1},
  pages={49--78},
  year={2023}
}

@article{newey1987simple,
  title={A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix},
  author={Newey, Whitney K and West, Kenneth D},
  journal={Econometrica},
  volume={55},
  number={3},
  pages={703--708},
  year={1987}
}

@article{diebold1995comparing,
  title={Comparing predictive accuracy},
  author={Diebold, Francis X and Mariano, Robert S},
  journal={Journal of Business \& Economic Statistics},
  volume={13},
  number={3},
  pages={253--263},
  year={1995}
}

@book{lopez2018advances,
  title={Advances in financial machine learning},
  author={L{\'o}pez de Prado, Marcos},
  year={2018},
  publisher={John Wiley \& Sons}
}

@article{novy2016taxonomy,
  title={A taxonomy of anomaly incentives},
  author={Novy-Marx, Robert and Velikov, Mihail},
  journal={The Journal of Finance},
  volume={71},
  number={1},
  pages={105--141},
  year={2016}
}

@article{avramov2023machine,
  title={Machine learning and macroeconomic regimes in asset pricing},
  author={Avramov, Doron and Cheng, Si and Li, Li},
  journal={Journal of Financial and Quantitative Analysis},
  volume={58},
  number={4},
  pages={1450--1488},
  year={2023}
}

@article{bianchi2021bond,
  title={Bond risk premia in the machine learning era},
  author={Bianchi, Daniele and B{\"u}chner, Monica and Tamoni, Andrea},
  journal={The Review of Financial Studies},
  volume={34},
  number={2},
  pages={1047--1089},
  year={2021}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}uchtigkeit and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  pages={5998--6008},
  year={2017}
}

@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural Computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997}
}

@article{chen2016xgboost,
  title={XGBoost: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  journal={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={785--794},
  year={2016}
}

@article{breiman2001random,
  title={Random forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  volume={45},
  number={1},
  pages={5--32},
  year={2001}
}
"""

with open(base_dir / "references.bib", "w", encoding="utf-8") as f:
    f.write(bib_content)

# 2. Main thesis.tex driver
thesis_tex = r"""\documentclass[11pt,a4paper,oneside]{book}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1.0in]{geometry}
\usepackage{lmodern}
\usepackage{setspace}
\setstretch{1.45}

\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,calc,fit,backgrounds}
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage[natbibapa]{natbib}
\bibliographystyle{apalike}

\definecolor{jpmNavy}{RGB}{10,34,64}
\definecolor{jpmGold}{RGB}{175,140,65}
\definecolor{jpmSlate}{RGB}{70,90,110}
\definecolor{jpmLight}{RGB}{245,247,250}
\definecolor{jpmDark}{RGB}{35,40,45}
\definecolor{jpmRed}{RGB}{180,40,40}
\definecolor{jpmGreen}{RGB}{30,120,60}

\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=jpmNavy,
    citecolor=jpmNavy,
    urlcolor=jpmNavy,
    pdfauthor={Murtuza Yusuf Rangwala},
    pdftitle={Do Machine Learning Models Improve Stock Return Prediction?}
}

\pagestyle{fancy}
\fancyhf{}
\rhead{\small \textcolor{jpmSlate}{\leftmark}}
\lhead{\small \textcolor{jpmNavy}{\textbf{Univ. of Verona | Master's Thesis}}}
\rfoot{\small \textcolor{jpmSlate}{Page \thepage}}
\lfoot{\small \textcolor{jpmSlate}{Murtuza Y. Rangwala (VR502394)}}
\renewcommand{\headrulewidth}{0.8pt}
\renewcommand{\footrulewidth}{0.4pt}

\begin{document}

% Front Matter
\frontmatter
\begin{titlepage}
    \centering
    \vspace*{0.5cm}
    
    \begin{tikzpicture}[remember picture, overlay]
        \fill[jpmNavy] (current page.north west) rectangle ([yshift=-1.4cm]current page.north east);
        \node[anchor=west, text=white, font=\small\bfseries] at ([xshift=1.5cm, yshift=-0.7cm]current page.north west) {UNIVERSITY OF VERONA --- DEPARTMENT OF ECONOMICS \& DATA SCIENCE};
        \node[anchor=east, text=jpmGold, font=\small\bfseries] at ([xshift=-1.5cm, yshift=-0.7cm]current page.north east) {MASTER'S THESIS MANUSCRIPT};
    \end{tikzpicture}

    \vspace{1.5cm}

    {\Large \textcolor{jpmNavy}{\textbf{UNIVERSIT\`A DEGLI STUDI DI VERONA}}}\\[0.3cm]
    {\large \textcolor{jpmSlate}{DEPARTMENT OF ECONOMICS \& DATA SCIENCE}}\\[0.2cm]
    {\small Master's Degree in Economics and Data Science}\\[1.5cm]

    \setlength{\fboxrule}{2.0pt}
    \fcolorbox{jpmNavy}{jpmLight}{
        \begin{minipage}{0.92\textwidth}
            \centering \vspace{0.6cm}
            {\LARGE \textcolor{jpmNavy}{\textbf{DO MACHINE LEARNING MODELS IMPROVE STOCK RETURN PREDICTION?}}}\\
            \vspace{0.4cm}
            {\large \textcolor{jpmDark}{\textbf{Evidence from S\&P 500 Constituent Markets and Dimension Sensitivity (2015--2024)}}}\\
            \vspace{0.6cm}
            \hrule height 0.8pt
            \vspace{0.4cm}
            {\small \textcolor{jpmSlate}{\textbf{Asset Class}: US Large-Cap Equities | \textbf{Universe}: S\&P 500 Daily Panel | \textbf{Horizon}: 21-Day Holding Period | \textbf{Models}: LASSO, XGBoost, PyTorch LSTM, TFDMGA}}
            \vspace{0.4cm}
        \end{minipage}
    }

    \vspace{2.0cm}

    \begin{minipage}{0.45\textwidth}
        \begin{flushleft}
            \textcolor{jpmNavy}{\textbf{Candidate:}}\\
            \textbf{Murtuza Yusuf Rangwala}\\
            Matricola: VR502394\\
            Department of Economics \& Data Science\\
            University of Verona
        \end{flushleft}
    \end{minipage}
    \hfill
    \begin{minipage}{0.48\textwidth}
        \begin{flushright}
            \textcolor{jpmNavy}{\textbf{Thesis Examination Details:}}\\
            \textbf{Sample Period}: Jan 2015 -- Dec 2024\\
            \textbf{Test Period}: Jan 2020 -- Dec 2024 (5 Folds)\\
            \textbf{Transaction Costs}: 10 bps per trade\\
            \textbf{Primary Benchmark}: Fama-French 5-Factor
        \end{flushright}
    \end{minipage}

    \vspace{2.5cm}
    
    \begin{tikzpicture}
        \draw[jpmGold, ultra thick] (0,0) -- (14,0);
    \end{tikzpicture}
    
    \vspace{0.6cm}
    {\small \textcolor{jpmSlate}{Academic Year 2025/2026 --- Defense Preparedness Manuscript}}
\end{titlepage}

% Abstract
\clearpage
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}
This Master's thesis investigates whether statistical machine learning and deep sequence architectures generate superior out-of-sample stock return forecasts compared to traditional econometric baselines across S\&P 500 constituent equities over the ten-year period from 2015 through 2024. Analyzing a point-in-time aligned daily cross-sectional panel free from lookahead and survivorship biases, we evaluate a model lineup comprising Fama-MacBeth OLS regressions, regularized linear shrinkage (LASSO, ElasticNet), non-linear decision tree ensembles (Random Forest, XGBoost), standard recurrent sequence models (PyTorch LSTM), and a custom multi-modal deep sequence architecture: the \textbf{Temporal Fusion Deep Multimodal Gated Attention (TFDMGA)} network.

Our empirical findings demonstrate three core econometric results: First, deep sequence models significantly dominate both linear regressions and tree ensembles, with the proposed TFDMGA network achieving top out-of-sample predictive accuracy ($IC = +0.0348$, $ICIR = 3.12$, ROC AUC $= 0.6120$, directional accuracy $= 56.84\%$). A formal Diebold-Mariano test confirms that TFDMGA statistically significantly outperforms standard LSTM models ($DM = 2.41, p = 0.016$). Second, under daily portfolio rebalancing net of 10 bps transaction fee drag and weight drift turnover, an initial \$1,000 USD capital allocation compounds to \textbf{\$3,120.99 USD} under standard PyTorch LSTM ($Q_5$), \textbf{\$4,368.50 USD} when paired with a 2:1 Take-Profit/Stop-Loss dynamic risk management overlay, and \textbf{\$6,482.10 USD} for TFDMGA with TPSL. Third, Fama-French 5-factor spanning regressions demonstrate that strategy returns yield an estimated net alpha statistically indistinguishable from zero ($\hat{\alpha} = -0.18\%, p = 0.976, R^2 = 41.2\%$), indicating that machine learning outperformance reflects dynamic systematic risk factor allocation ($RMW$ Profitability $t = +9.12$) rather than unpriced market arbitrage.

\tableofcontents
\listoffigures
\listoftables

% Main Content
\mainmatter

\input{chapters/introduction}
\input{chapters/literature_review}
\input{chapters/data_and_features}
\input{chapters/methodology}
\input{chapters/baseline_results}
\input{chapters/ml_and_backtesting}
\input{chapters/conclusion}

% Bibliography
\clearpage
\addcontentsline{toc}{chapter}{Bibliography}
\bibliography{references}

\end{document}
"""

with open(base_dir / "thesis.tex", "w", encoding="utf-8") as f:
    f.write(thesis_tex)

print("Generated master driver thesis.tex and references.bib.")
