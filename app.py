import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Will This Visitor Buy?", page_icon="🛒", layout="centered")

@st.cache_resource
def load_artifact():
    path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

try:
    art = load_artifact()
except FileNotFoundError:
    st.error("Model file `model.joblib` is missing. Run all cells in `purchase_intent.ipynb` from this project folder first.")
    st.stop()

pipe, thr, sure = art["pipeline"], art["threshold"], art["sure_buy"]
defaults = art["defaults"]

st.title("🛒 Will This Visitor Buy?")
st.caption("Real-time purchase-intent prediction for e-commerce sessions. Enter what the visitor did on the site and get a suggested action.")

tab_pred, tab_about = st.tabs(["Predict", "About the model"])

with tab_pred:
    st.subheader("Session details")
    c1, c2 = st.columns(2)
    with c1:
        admin = st.number_input("Account / admin pages viewed", 0, 100, 0)
        admin_d = st.number_input("Time on admin pages (seconds)", 0.0, 20000.0, 0.0)
        info = st.number_input("Info pages viewed", 0, 100, 0)
        info_d = st.number_input("Time on info pages (seconds)", 0.0, 20000.0, 0.0)
        prod = st.number_input("Product pages viewed", 0, 1000, 10)
        prod_d = st.number_input("Time on product pages (seconds)", 0.0, 100000.0, 300.0)
    with c2:
        bounce = st.slider("Bounce rate", 0.0, 0.2, 0.02, 0.001, help="Share of visits to these pages that left right away (0 to 0.2).")
        exit_r = st.slider("Exit rate", 0.0, 0.2, 0.04, 0.001, help="Share of views of these pages that were the last page of a visit.")
        page_val = st.number_input("Page value", 0.0, 500.0, 0.0, help="Average value of the pages visited before a purchase (from your analytics tool).")
        month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
        visitor = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = st.checkbox("Weekend session")
        special = st.select_slider("Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)

    with st.expander("Advanced (technical details, defaults are fine)"):
        osys = st.number_input("Operating system code", 1, 20, defaults["OperatingSystems"])
        browser = st.number_input("Browser code", 1, 30, defaults["Browser"])
        region = st.number_input("Region code", 1, 20, defaults["Region"])
        traffic = st.number_input("Traffic type code", 1, 30, defaults["TrafficType"])

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "Administrative": admin, "Administrative_Duration": admin_d,
            "Informational": info, "Informational_Duration": info_d,
            "ProductRelated": prod, "ProductRelated_Duration": prod_d,
            "BounceRates": bounce, "ExitRates": exit_r, "PageValues": page_val,
            "SpecialDay": special, "Month": month, "OperatingSystems": osys,
            "Browser": browser, "Region": region, "TrafficType": traffic,
            "VisitorType": visitor, "Weekend": int(weekend),
        }])
        p = float(pipe.predict_proba(row)[0, 1])

        st.metric("Purchase probability", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= sure:
            st.success("**Suggested action: no action.** This visitor is very likely to buy anyway, so a discount would be wasted.")
        elif p >= thr:
            st.warning("**Suggested action: show a coupon.** This visitor is on the fence, so a small nudge can tip the decision.")
        else:
            st.info("**Suggested action: show the chat assistant.** Low purchase intent, so offer help instead of a discount.")
        st.caption("This is a decision-support estimate, not a guarantee.")

with tab_about:
    m = art["metrics"]
    st.write(f"**Model:** {art['model_name']} (tuned with cross-validation, evaluated on a held-out test set).")
    st.write(f"**Decision threshold:** {thr:.2f} (chosen on training data to favour recall).")
    st.table(pd.DataFrame({"Metric": ["Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"],
                           "Test score": [f"{m['recall']:.3f}", f"{m['precision']:.3f}", f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}", f"{m['pr_auc']:.3f}"]}))
    st.write("**Actions:** below the threshold → chat assistant · between the threshold and "
             f"{sure:.0%} → coupon · above {sure:.0%} → no action.")
    st.write("**Limitations:** trained on one retailer's sessions; predictions improve as the session progresses; the coupon effect should be confirmed with an A/B test.")
    st.caption("Data: UCI Online Shoppers Purchasing Intention (Sakar & Kastro, 2018).")
