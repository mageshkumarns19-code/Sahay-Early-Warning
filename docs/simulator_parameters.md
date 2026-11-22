# Simulation parameters and sources

Tags: **S** = taken from a cited source · **S~** = derived from a cited source with an approximation · **A** = modelling assumption (no published Indian figure), varied in the sensitivity analysis.

| Parameter | Value used | Tag | Source / basis |
|---|---|---|---|
| Offence mix | grievous hurt 35%, witness intimidation 9%, rape/gang rape 14%, murder of a family member 2.5%, arson 0.5%, other caste violence 39% | S~ | NCRB *Crime in India* 2022–2023 (simple hurt 32.0%, criminal intimidation 9.2%; rape 13.9%, murder 2.4%, arson 0.2% from Table 7B.5, metropolitan cities). Arson raised to 0.5% so the group exists. To be confirmed against all-India Table 7A.1. |
| Language mix | Hindi 70%, Telugu 7%, Marathi 5%, Odia 5%, Kannada 4%, Tamil 4%, Gujarati 2%, Punjabi 2%, Bengali 1% | S~ | NCRB 2023 State shares (57,789 crimes; Uttar Pradesh 26%; about two-thirds in four Hindi-speaking States) |
| Distress trajectories | resilient 62.0%, recovering 19.6%, chronic 10.0%, delayed 8.4% | S | Galatzer-Levy, Huang & Bonanno (2018), *Clinical Psychology Review* 63:41–55 (pooled 65.7/20.8/10.6/8.9%, normalised) |
| Charge-sheet timing | week 4–26 | S + S~ | PoA Rules, Rule 7(2): within 60 days; NCRB 2023: 14,634 cases pending investigation at year end |
| Charge-sheeting rate | 85% | S | NCRB 2023: 45,439 of 53,372 PoA cases (85.1%) |
| Trial outcome within 24 months | none simulated | S | Court pendency of crimes against Scheduled Castes above 90% |
| Interval between hearings | mean about 10 weeks | A | Tested at about 6 and 16 weeks |
| Relief instalments | due in week 1 and at charge sheet | S | Rule 12(4) and Annexure I |
| Relief paid on time / delay | 30% on time; mean delay 12 weeks | A | Tested at 70%/4 weeks and 10%/24 weeks |
| Bail of the accused | 55% | A | Tested at 30% |
| Effects of legal events on distress | hearing (anticipatory and on the day), bail, threat, relocation, overdue relief | A | Informed by the secondary-victimisation literature; tested at ×0.5 and ×2 |
| Check-in channel, age mix | voice call 50%, chatbot 30%, app 20% | A | Design choice |

All values are defined in the `P` dictionary at the top of `src/sahay_modelC.py` (section 1).
