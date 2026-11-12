# Executive summary

**Which options respond to which macro news, and do they charge the right amount?**

Erdős Institute, Introduction to Quantitative Methods in Finance, Fall 2026.

## The problem

Several times a month the US government publishes a number that can move markets: the inflation rate (CPI), the jobs report, GDP and others. These releases come out at 8:30 in the morning on dates that are known weeks ahead. That makes them unusual among market risks. Nobody knows what the number will be, but everybody knows exactly when it will arrive.

Options are the instruments that put a price on that kind of risk. An option that expires tomorrow is worth more if tomorrow is likely to be a big day. So on the evening before a release, the price of one-day options contains the market's answer to a specific question: how much extra movement will this release cause?

This project asks whether that answer was right. It has three parts.

1. **How do stocks actually respond to each kind of news?** A release matters to stocks largely through interest rates. News that makes higher rates more likely pushes bond prices down, and stocks may or may not follow. I measure how much stocks move per unit of bond move, for each type of release, and how that changed between 2022 and 2026.
2. **What do options charge for each release?** I measure the extra amount built into option prices the evening before, for each type of release, for options of different lengths, and for options on large companies (SPY), technology companies (QQQ) and small companies (IWM).
3. **Was the charge right?** If options charge too much for a release, someone who sells them and hedges will make money on average; if too little, they will lose. I run that trade on every evening in the sample using the pricing and hedging methods taught in the course.

For the inflation release there is also an independent check. Kalshi, a regulated prediction market, lets people bet directly on what the inflation number will be. Its prices show how uncertain the market is about the number itself, which separates two things an option price mixes together: how big the news might be, and how strongly stocks react to it.

## What was measured, in plain terms

- **Release morning and quiet day.** A release morning has one of seven scheduled 8:30 releases. A quiet day has none of them and no Federal Reserve meeting. Quiet days are the benchmark for everything: about 660 of them, against about 450 release mornings, from May 2022 to October 2026.
- **The stock–bond link (γ).** In the 15 minutes after a release, how much does the stock index move for each 1% move in long-term Treasury bonds caused by the news? A value near 1 means stocks fall about as much as bonds on bad rate news. A value near 0 means bonds move and stocks do not. It is estimated by comparing release mornings with quiet mornings, a standard method due to Rigobon and Sack (2004).
- **The event premium.** Option prices can be converted into the amount of movement they imply for the next day (the implied variance). The event premium is the implied variance on the evening before a release, minus its level on nearby quiet evenings.
- **The hedged straddle.** A straddle is a call and a put bought or sold together, which pays off when the market moves a lot in either direction. Selling one, and holding shares to cancel out the direction of the move, leaves a pure bet that the market will move less than options implied. Its average profit is the test of whether options overcharged. It is built from the Black–Scholes price and delta from the course, and checked against simulated data before being used on real prices.

## Findings

**1. In 2022, rate news hurt stocks whatever its source. That link has since more than halved.** In 2022–23 the stock index moved 0.86% for every 1% move in bonds on release mornings. In 2024–26 it moved 0.35%. The change is statistically clear, and it happened between 2022 and 2023.

**2. It now depends on what kind of news it is.** Since 2024, inflation releases still pull stocks down with bonds (0.66% per 1%). Releases about economic activity, such as the jobs report, do not (0.05%). On jobs mornings bonds still move as much as before, but stocks no longer follow. The natural reading is that a strong jobs number used to mean "the Fed will raise rates" and now also means "the economy is healthy".

**3. Options treat four releases as events, and repriced them as the regime changed.** Inflation, Federal Reserve meetings, the jobs report and the PCE inflation measure carry a clear premium in one-day option prices. GDP, retail sales, producer prices and jobless claims carry none. The premium for the inflation release fell about fivefold between 2022–23 and 2024–26; the premium for the jobs report fell by only a third. Inflation went from by far the most expensive release to about level with jobs.

**4. Options charged too little for release days in 2022–23.** Selling one-day options before a release earned 17 basis points (0.17% of the index) less than selling before a quiet day, a statistically strong gap. Release days delivered 95% of the movement options had priced; quiet days delivered only 77%. By 2024–26 the gap had disappeared. The market under-priced scheduled news during the inflation shock and corrected within about two years.

**5. Selling one-day options is not a free lunch.** The widely quoted one-day volatility index (VIX1D) is about 6% higher than the volatility at which at-the-money options actually trade. Using it to price the trade produces an apparent steady profit of 8 to 14 basis points a day. At real traded prices, available from 2024, the profit is 2 to 4 basis points, not distinguishable from zero, before trading costs.

**6. Small-company options carry the largest premium for macro news.** Measured against each market's normal day, options on small companies price inflation, jobs and Fed days about twice as heavily as options on large companies.

**7. The prediction market confirms the option pricing for inflation.** In 2022–23, option prices and Kalshi together implied that stocks would move 9.3% per percentage point of inflation surprise; they actually moved 9.8%. In 2024–26 the reaction fell to 3.8% and option prices are consistent with it. An early result suggesting options overcharged by a factor of two to three did not survive closer testing, and a trading rule based on Kalshi made no money.

## What testing ruled out

Four results looked convincing at some stage and were withdrawn after a check. An apparent overpricing of inflation risk by a factor of two to three vanished once the stock sensitivity was allowed to change over time, and it also appeared on randomly chosen days with no release. A claim that stocks had stopped reacting to inflation surprises came from measuring overnight returns, where unrelated noise hides a reaction that is clear in the first 15 minutes. The steady profit from selling one-day options came from using an index in place of traded prices. A large profit from selling small-company options before Federal Reserve meetings came from a data filter that had dropped nine of the 21 meetings. The findings above are the ones that remained.

## What this is worth to a practitioner

- A portfolio manager gets a current, measured answer to "how much does this release matter for my stocks?" It is less than half what it was in 2022, and for activity data close to nothing.
- An options desk gets a ranking of which releases the market pays for, a check that those prices have matched outcomes since 2024, and a warning that VIX1D overstates what can be earned.
- The 2022–23 result shows that even a heavily traded options market can take a year or two to price a new regime.

## Limitations

- The sample is four and a half years with one change of regime. Results for 2022–23 rest on 174 release days.
- Real traded option prices are only available from February 2024. The finding that options were too cheap before releases in 2022–23 rests on the VIX1D index, compared between release and quiet evenings so that its known bias cancels.
- Option prices are last trades, without the cost of the bid-ask spread. No number here is a net trading profit.
- Many combinations of release and market were examined. Only the pooled results above are claimed; individual cells are leads.
- One week (3 to 11 April 2025, the tariff announcement) was excluded from the option tables after seeing that it distorted them.
- The prediction-market check covers inflation only, with 52 releases.

## Next steps

- Obtain historical option quotes for 2022–23 to confirm finding 4 at traded prices.
- Add trading costs, and test the small-company result on new data.
- Extend the prediction-market check to the jobs report and other releases as Kalshi's history grows.
