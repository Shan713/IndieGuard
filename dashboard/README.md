# Dashboard

The interactive dashboard is live at **https://indieguard.streamlit.app**.

Its code is in [`src/dashboard/`](../src/dashboard) (kept there because the deployed app on Streamlit Community Cloud runs
`src/dashboard/app.py` from this repository). Run it locally from the repository root:

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

Pages: market explorer, tag rules and complaint themes, patch impact, launch-risk scorer (with reasons and suggested changes), about.
Details: [`docs/dashboard.md`](../docs/dashboard.md).
