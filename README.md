<div align="center">

<!-- Adjusted SVG Header: Font size 22px, Width 800px -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=3B82F6&center=true&vCenter=true&width=800&height=60&lines=Multi-Class+Match+Result+Prediction;Multinomial+Logistic+Regression+Engine;Probabilistic+Outcome+Modeling+(Win%2FLoss%2FDraw)" alt="Typing SVG" />
</a>

<p align="center">
  <b>A mathematical and computational framework converting raw match metrics into calibrated class probabilities.</b>
</p>

<!-- Badge Matrix -->
<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://sqlite.org/"><img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"></a>
  <a href="#academic-context"><img src="https://img.shields.io/badge/KLH--University-25SC2107E-8A2BE2?style=for-the-badge" alt="Course"></a>
</p>

---

<!-- Visual Metric Highlights -->
<table>
  <tr>
    <td align="center" width="25%"><b>Algorithm</b><br><code>Multinomial LR</code></td>
    <td align="center" width="25%"><b>Outcome Classes</b><br><code>3 (Win/Loss/Draw)</code></td>
    <td align="center" width="25%"><b>Loss Function</b><br><code>Cross-Entropy</code></td>
    <td align="center" width="25%"><b>Deployment</b><br><code>Streamlit / Flask</code></td>
  </tr>
</table>

</div>

<br>

> [!IMPORTANT]
> **Deterministic vs. Probabilistic Modeling:** Traditional sports prediction models often force binary outputs (Win/Loss), ignoring the high statistical likelihood of draws in low-scoring sports. This system models full multinomial probability distributions $P(Y=k \mid \mathbf{x})$, retaining draw probability mechanics.

---

## 👁️ System Architecture

```mermaid
graph TD
    A["Kaggle / SQLite Historical Data"] --> B["Feature Extraction & Standardizer"]
    
    subgraph Pipeline Processing Engine
        B --> C1["Home / Away Advantage Factor"]
        B --> C2["Recent Form Momentum Index"]
        B --> C3["Head-to-Head Ratio Metrics"]
    end

    C1 --> D["Softmax Linear Layer"]
    C2 --> D
    C3 --> D
    
    D --> E{"Multinomial Estimator"}
    
    E --> F1["Class 0: Win"]
    E --> F2["Class 1: Draw"]
    E --> F3["Class 2: Loss"]

    F1 --> G["Streamlit Interactive Interface"]
    F2 --> G
    F3 --> G

    classDef data fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef proc fill:#111827,stroke:#10b981,stroke-width:2px,color:#fff;
    classDef model fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#fff;
    classDef out fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#fff;

    class A,G data;
    class B,C1,C2,C3 proc;
    class D,E model;
    class F1,F2,F3 out;
