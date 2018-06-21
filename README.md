# Quantopian-Project
Computational Investing F2017

# Import the libraries we will use here.
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.factors import CustomFactor
import quantopian.optimize as opt
import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import pandas as pd
import math

def initialize(context):
    
    #Called once at the start of the program. Any one-time
    #startup logic goes here.
    
   
    #Implement short and long leverage factors.
    use_beta_hedging = True
    context.long_leverage = 1.8
    context.short_leverage = -0.6
    
   
    
                     
    #Rebalance weekly on first day of week at market open
    schedule_function(rebalance,
                      date_rule=date_rules.week_start(),
                      time_rule=time_rules.market_open())
    if use_beta_hedging:
        # Call the beta hedging function one hour later to
        # make sure all of our orders have gone through.
        schedule_function(hedge_portfolio,
                          date_rule=date_rules.week_start(),
                          time_rule=time_rules.market_open())
    # trading days used for beta calculation
    context.lookback = 365
    # Used to aviod purchasing any leveraged ETFs 
    context.dont_buys = security_lists.leveraged_etf_list
    # Current allocation per asset
    context.pct_per_asset = 0
    context.index = symbol('SPY')
    
    
   
    
    

    # Create and attach our pipeline (dynamic stock selector), defined below.
    attach_pipeline(make_pipeline(context), 'momentum_test')
    
    
class StdDev(CustomFactor):
    def compute(self, today, asset_ids, out, values):
        # Calculates the column-wise standard deviation, ignoring NaNs
        out[:] = np.nanstd(values, axis=0)
        
class SMA_200_Yesterday(CustomFactor):  
    inputs = [USEquityPricing.close]  
    # set the window length to 1 plus the moving average window  
    window_length = 100

    def compute(self, today, assets, out, close):  
    # Take the mean of previous days but exclude the latest date  
        out[:] = np.nanmean(close[0:-1], axis=0)


class PriceHistorical(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 1
    def compute(self,today,assets,out,close):
        out[:] = close[0]
        
def make_pipeline(context):

    universe = Q1500US()



    # Create a dollar_volume factor using default inputs and window_length
    dollar_volume = AverageDollarVolume(window_length=1,mask = universe)
    
    # Define high dollar-volume filter to be the top 50% of stocks by dollar volume.
    high_dollar_volume = dollar_volume.percentile_between(50, 100)
    context.assets = high_dollar_volume
    
    
    
    #Latest p/e ratio
    pe_ratio = morningstar.valuation_ratios.pe_ratio.latest
    
    #Returns over the past 30 days  
    recent_returns = Returns(window_length=30,
                             mask=high_dollar_volume)
    #Price Momentum
    current_price = USEquityPricing.close
    #Lookback window for price momentum
    n = 400
    #Price Momentum
    current_price = USEquityPricing.close.latest
    historical_price = PriceHistorical(window_length = n)
    
    total_return_momentum = current_price/historical_price - 1
    #SMA Differential
    sma_differential = (SimpleMovingAverage(inputs = [USEquityPricing.close],window_length = n/10)/SimpleMovingAverage(inputs = [USEquityPricing.close],window_length = n)) - 1
    #Price to SMA Differential
    current_sma = SimpleMovingAverage(inputs = [USEquityPricing.close],window_length = n)
    price_sma_diff = (current_price / SimpleMovingAverage(inputs = [USEquityPricing.close],window_length = n)) - 1
    #Instant SMA Slope
    sma_200_yesterday = SMA_200_Yesterday()
    instant_sma = current_sma/sma_200_yesterday - 1
                           
    std_dev = StdDev(
    inputs = [USEquityPricing.close],
    window_length = n
        )
    z_score = (current_price - current_sma)/std_dev
    
    #Overall score of price momentum based on 5 price momentum indicators
    #Instant_sma weighted higher because it exhibits the highest returns when used on its own.
    price_score = .2*(total_return_momentum) + .1*(sma_differential) + .2*(price_sma_diff) + .4*(instant_sma) + .1*(z_score)
    
    
    
    #Top 5% and bottom 5% of security returns in high dollar-volume group.
    high_returns = recent_returns.percentile_between(96,100)
    low_returns = recent_returns.percentile_between(0,5)

    # Top and bottom percentiles of PE ratio
    top_pe_stocks = pe_ratio.percentile_between(97, 100, mask = high_dollar_volume)
    bottom_pe_stocks = pe_ratio.percentile_between(0, 6, mask = high_dollar_volume)
    
    # Top and bottom percentiles of price momentum score
    good_price = price_score.percentile_between(90,100, mask = high_dollar_volume)
    poor_price = price_score.percentile_between(0,10, mask = high_dollar_volume)
    
    # Defines which stocks we are longing and shorting
    longs =  good_price and top_pe_stocks and high_returns
    shorts = poor_price and bottom_pe_stocks and low_returns
   
    # Final breakdown of the filtered securities we are trading.
    securities_to_trade = (longs | shorts )
  
    # Creates our pipeline
    pipe = Pipeline(
        columns={
            'longs' : longs,
            'shorts': shorts,
             
      },
      screen = securities_to_trade
    )

    return pipe



        
def before_trading_start(context, data):
    """
    Called every day before market open. This is where we get the securities
    that made it through the pipeline.
    """

    # Pipeline_output returns a pandas DataFrame with the results of our factors
    # and filters.
    context.output = pipeline_output('momentum_test')

    
    
    
    #context.returns_sec = context.output[context.output['high_returns']]
    
   
    context.longs = context.output[context.output['longs']]
    
    context.shorts = context.output[context.output['shorts']]
    
    # A list of the securities that we want to order today.
    #context.longs_list = context.longs.index.tolist()


    # A list of the securities that we want to order today.
    #context.shorts_list = context.shorts.index.tolist()
     
    #List of securities we want to order today
    context.security_list = context.longs.index.union(context.shorts.index).tolist()

    # A set of the same securities, sets have faster lookup.
    context.security_set = set(context.security_list)

    
def compute_weights(context):
    """
    Compute weights to our long and short target positions.
    """
    # Set the allocations to even weights for each long position
    long_weight = context.long_leverage / len(context.longs)
    short_weight = context.short_leverage/ len(context.shorts)
    
    return long_weight, short_weight

def compute_weights_quarter(context):
    
    long_weight_q = context.long_leverage_quarter/len(context.longs)
    short_weight_q = context.short_leverage_quarter/len(context.shorts)
    
    return long_weight_q, short_weight_q


def hedge_portfolio(context, data):
    """
    This function places an order for "context.index" in the 
    amount required to neutralize the beta exposure of the portfolio.
    Note that additional leverage in the account is taken on, however,
    net market exposure is reduced.
    """
    factors = get_alphas_and_betas(context, data)
    beta_exposure = 0.0
    count = 1
    for asset in context.portfolio.positions:
        if asset in factors and asset != context.index:
            if not np.isnan(factors[asset].beta):
                beta_exposure += factors[asset].beta
                count += 1
    beta_hedge = -.8 * beta_exposure / count
    dollar_amount = context.portfolio.portfolio_value * beta_hedge
    record(beta_hedge=beta_hedge)
    if not np.isnan(dollar_amount):
        order_target_value(context.index, dollar_amount)
    
def get_alphas_and_betas(context, data):
    """
    returns a dataframe of 'alpha' and 'beta' exposures 
    for each asset in the current universe.
    """
    prices = history(context.lookback, '1d', 'price', ffill=True)
    returns = prices.pct_change()[1:]
    index_returns = returns[context.index]
    factors = {}
    for asset in context.portfolio.positions:
        try:
            y = returns[asset]
            factors[asset] = linreg(index_returns, y)
        except:
            log.warn("[Failed Beta Calculation] asset = %s"%asset.symbol)
    return pd.DataFrame(factors, index=['alpha', 'beta'])
    
def linreg(x, y):
    # We add a constant so that we can also fit an intercept (alpha) to the model
    # This just adds a column of 1s to our data
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.params[0], model.params[1]

def rebalance(context,data):
    """
    This rebalancing function is called according to our schedule_function settings.
    """
    #get the long and short weights
    long_weight , short_weight = compute_weights(context)
    
    #Orders the stocks in our security list according to our longs and shorts lists
    for stock in context.security_list:
        if data.can_trade(stock):
            if stock in context.longs.index:
                order_target_percent(stock, long_weight)
            elif stock in context.shorts.index:
                order_target_percent(stock, short_weight)
                
    #Sells stocks in our portfolio that are no longer in our security list
    for stock in context.portfolio.positions:
        if stock not in context.security_set and data.can_trade(stock):
            order_target_percent(stock, 0)
    
    # Log the long and short orders each week.
    log.info("This week's longs: "+", ".join([long_.symbol for long_ in context.longs.index]))
    log.info("This week's shorts: "  +", ".join([short_.symbol for short_ in context.shorts.index])) 
    
    
    
def trade(context, data):  
    context.counter += 1  
    if context.counter == 3:  
        q_rebalance()  
    context.counter = 0 

def q_rebalance(context,data):
    long_weight_q , short_weight_q = compute_weights_quarter(context)
    
    for stock in context.security_list:
        if data.can_trade(stock):
            if stock in context.longs.index:
                order_target_percent(stock, long_weight_q)
            elif stock in context.shorts.index:
                order_target_percent(stock, short_weight_q)
                
    for stock in context.portfolio.positions:
        if stock not in context.security_set and data.can_trade(stock):
            order_target_percent(stock, 0) 
                
    log.info("This week's longs: "+", ".join([long_.symbol for long_ in context.longs.index]))
    log.info("This week's shorts: "  +", ".join([short_.symbol for short_ in context.shorts.index])) 
    
def record_vars(context, data):
    """
    This function is called at the end of each day and plots certain variables.
    """

    # Check how many long and short positions we have.
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        if position.amount < 0:
            shorts += 1

    # Record and plot the leverage of our portfolio over time as well as the
    # number of long and short positions. Even in minute mode, only the end-of-day
    # leverage is plotted.
    record(leverage = context.account.leverage, long_count=longs, short_count=shorts)
