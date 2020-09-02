'''
/****************************************************************************************************/
/*  # Exchange Risk and IRR                                                                         */
/*  #Author: Rui Lima                                                                               */

     Tested with Python3
*/
/****************************************************************************************************/
'''
#----------------------------------------IMPORT LIBRARIES------------------------------------------
import pandas as pd
from datetime import date
from math import sqrt
from numpy_financial import irr
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------ASSUMPTIONS----------------------------------------------
spot_GBPUSD = 1.2401                  # GBPUSD spot
vol_GBPUSD = 0.1228                   # Implied Volatility GBPUSD
MCS_num = 1000                        # Number of MC simulations
start = date(2020,6,30)               # current date
percentiles = np.array([5,50,95])     # IRR Percentiles to analyse

#-------------------------------------------FUNCTIONS------------------------------------------------
# Function to calculate Geometric Brownian Motion without drift
def GBM(spot,vol,time,rand=[]):
    # rand - uniform distribution [0,1]
    # norm.ppf - inverse cumulative standard normal
    return spot * np.exp(-0.5 * vol ** 2 * time + vol * sqrt(time) * norm.ppf(rand))

#------------------------------------------IMPORT DATA-----------------------------------------------
# Import relevant columns from Excel spreadsheet
df = pd.read_excel('Cashflow GBP data.xlsx',parse_dates=True,usecols=[0,4,5,6])
# Simplify column names
df.columns=('Date','CF Local','Local Currency','Fund Currency')
# Calculate base internal rate return (GBP)
irr_base_case = irr(df['CF Local'])

#-------------------------------------MONTE CARLO SIMULATION------------------------------------------
# List with cash flow periods (years)
period=[df.iloc[i,0].year-start.year for i in range(len(df))]
# Initialise dataframe for GBPUSD rates in each simulation and time period corresponding to cashflow
df_fx = pd.DataFrame(np.zeros((len(period), MCS_num)),columns=list(range(1,MCS_num+1)))
# Set seed for replication
np.random.seed(1)
# Calculate GBPUSD for each simulation and time period
for i,row in df_fx.iterrows():
    if i == 0:
        row[:] = spot_GBPUSD
    else:
        rand_unif = np.random.uniform(0,1,MCS_num)
        row[:]=GBM(df_fx.iloc[i-1,:],vol_GBPUSD,(period[i]-period[i-1]),rand_unif)

print('\n-----------------------GBPUSD spot---------------------')
print(df_fx)

# Calculate USD cash flow (GBPUSD * Cash Flow)
df_CF_USD = df_fx.multiply(df['CF Local'],axis='index')
print('\n-----------------------USD CASH FLOW---------------------')
print(df_CF_USD)

#-------------------------------------INTEREST RATE OF RETURN-----------------------------------------
# Calculate IRR (%) USD for each simulation
irr_USD = [irr(df_CF_USD[i])*100 for i in range(i,MCS_num+1)]
# Calculate IRR percentiles
percentiles_irr_USD = np.percentile(irr_USD,percentiles)
print('\n----------------Percentiles------------------')
for i in range(len(percentiles)):
    print('{0:s}% percentile is: {1:s}%'.format(str(percentiles[i]),str(round(percentiles_irr_USD[i],3))))

#----------------------------------------------PLOTS--------------------------------------------------
ax = sns.distplot(irr_USD,bins=round(sqrt(MCS_num)),kde=False,axlabel='IRR in USD (%)')
ax.axvline(percentiles_irr_USD[0],color='grey',ls='--',label='5%   = {:.3f}%'.format(percentiles_irr_USD[0]))
ax.axvline(percentiles_irr_USD[1],color='grey',ls='--',label='50% = {:.3f}%'.format(percentiles_irr_USD[1]))
ax.axvline(percentiles_irr_USD[2],color='grey',ls='--',label='95% = {:.3f}%'.format(percentiles_irr_USD[2]))
plt.title('Interest Rate of Return (IRR) in fund currency (USD)')
plt.legend(fontsize=8)
plt.show()