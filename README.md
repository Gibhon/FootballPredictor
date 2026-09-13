### Model Evaluation v1.0
* **Predictive Behavior:** The model rarely predicts draw via argmax, but assigns plausible probability mass (~20–30%) to draw outcomes consistent with the base rate.
* **Evaluation Choice:** Evaluated via log-loss and Brier score rather than accuracy for this reason.
* **Takeaway:** Probabilistic scoring metrics ensure the model is rewarded for accurate uncertainty calibration on draws, which a rigid metric like raw accuracy would misrepresent.


### v2.0 Updates
* **Features Added:** Draw rate, goal difference, and home/away-split rolling form.
* **Performance Impact:** Negligible improvement (log-loss 1.016 → 1.014, Brier 0.608 → 0.607).
* **Takeaway:** This confirms that rolling-average features are near saturated. Real predictive gains will likely require pivoting to fundamentally different signal types, such as Elo ratings or head-to-head records, rather than piling on similar metrics.