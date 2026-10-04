# Calibration notes (read before reusing the data)
The inputs are synthetic by design. To keep the report's published statistics reproducible, the generator parameters were fitted once and then frozen:
1. **Supplier panel** - five risk attributes are generated from a shared latent factor, then affinely mapped so mean, SD, min and max match Table 5; the seed (651) was chosen so the K-Means result has 6 elevated-risk suppliers and PCA variance of 47% / 22%.
2. **Demand history** - trend x fiscal-month seasonality x log-normal noise; seed and growth/noise parameters were fitted so the holdout MAPE of damped Holt-Winters is 5.1% and FY28/FY27 growth is about 2.9%. The FY27 forecast is scaled to 58.86 GW (the published figure).
3. **Scheduling instance** - order-book shape, opening domestic-cell stock, the FIFO mis-issue share and per-MW margins were fitted so the FIFO and LP schedules reproduce the Table 7 KPIs.
Because of this, the synthetic data demonstrate the *method*; they are not evidence about any real plant. Real deployment requires ERP/MES data.
