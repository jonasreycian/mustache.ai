from typing import Dict, List

class AverageCostMethod:
    def __init__(
        self,
        stock_name: str,
        initial_equity: int,
        capital_per_trade: int,
        capital_at_risk: int,
        max_hold_period: int,
        isBacktest: bool = False,
    ) -> None:
        self._stock_name = stock_name
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.port_gain_loss = 0
        self.port_gain_loss_pct = 0
        self.capital_per_trade = capital_per_trade
        self.cumm_gain = 0  # Cummulative gain
        self.cumm_loss = 0  # Cummulative loss
        self.cumm_profit = 0  # Cummulative loss
        self.total_trade = 0  # Total trades made
        self.win_count = 0  # Win count
        self.position_date = None
        self.capital_at_risk = capital_at_risk
        self.gain_loss = 0  # Gain/Loss of the current position
        self.gain_loss_pct = 0  # Gain/Loss Pct of the current position
        self.average_price = 0  # Average Price of share after PSE Charges
        self.average_value = 0  # Average Value of total share
        self.market_price = 0  # Current price of the stock
        self.market_value = 0  # Current market value (Selling PSE Charges applied)
        self.position_days = 0  # Holding period of the stock
        self.current_board_lot = 0  # Current board lot of the stock
        self.current_shares = 0  # Current shares
        self.current_date = None
        self.max_hold_period = max_hold_period  # Get the max hold period
        self.logs = List()
        self.isBacktest = isBacktest

    def __str__(self) -> str:
        str = 0
        return str

    def _take_action(self, action, market_price, current_date):
        """
        Strategies:
            1. Will use a fixed capital to trade per buy like tranching method
            2. When sell occurs, sell all positions and use accumulated_profit
        """

        self.market_price = market_price
        self.current_board_lot = self._getBoardLot(market_price)
        self.current_date = current_date

        # BUY Trigger
        if(action == 1):
            # No buy when out of buying power
            if self.current_equity < self.capital_per_trade:
                return self

            # Determine max share based on capital
            max_share = self.capital_per_trade / self.market_price
            max_share_per_lot = max_share - (max_share % self.current_board_lot)

            # Buying power can't accomodate the minimum slot
            if max_share_per_lot > 0:
                return self

            if(self.position_days == 0):
                self.position_days = 1
                self.position_date = current_date
            else:
                self.position_days += 1


            # Apply PSE Charges
            self._buy(market_price, max_share_per_lot)

        # Sell Trigger
        elif(action == 2):
            self._sell()

        # Hold Trigger
        else:
            # Compute the current market value
            self._getCurrentMarketValue()

            if(self.position_days > self.max_hold_period or self.port_gain_loss_pct >= self.capital_at_risk):
                self._sell()

            self.position_days += 1

    def _buy(self, market_price, max_share_per_lot) -> None:
        """
        Apply PSE Charges when buy
        """

        # PSE Charges
        #   Gross Transaction Amount    : share x price
        #   Broker's Commission         : 0.25% x GROSS_TRANSACTION_AMOUNT
        #   Broker's Commission VAT     : 12% x BROKER_COMMISION
        #   SCCP Fee                    : 0.01% x GROSS_TRANSACTION_AMOUNT
        #   PSE Transaction Fee         : 0.0005% x GROSS_TRANSACTION_AMOUNT

        gross_transaction_amount = market_price * self.current_shares
        brokers_commision = 0.0025 * gross_transaction_amount
        brokers_commision = 20 if brokers_commision <= 20 else brokers_commision

        brokers_commision_vat = 0.12 * brokers_commision
        sccp_fee = 0.0001 * gross_transaction_amount
        pse_transaction_fee = 0.00005 * gross_transaction_amount

        net_transaction_amount = (
            gross_transaction_amount
            + brokers_commision
            + brokers_commision_vat
            + sccp_fee
            + pse_transaction_fee
        )

        # Use Averaging Cost method (Total Cost / Total Shares) when tranching
        if self.current_shares > 0:
            self.average_value += net_transaction_amount
            self.average_price = self.average_value / self.current_shares
        else:
            self.average_price = net_transaction_amount / max_share_per_lot
            self.average_value = self.average_price + self.current_shares

        # Update the current equity
        self.current_shares += max_share_per_lot
        self.current_equity -= net_transaction_amount

        # Compute the current market value
        self._getCurrentMarketValue()

        if self.isBacktest:
            # Record the trade made
            self.logs.append(
                self.current_date,
                self.current_equity,
                self.stock_name,
                self.market_price,
                self.average_price,
                self.current_shares,
                self.market_value,
                self.gain_loss,
                self.gain_loss_pct,
                "BUY",
                f"{self.gain_loss_pct % 100}%",
            )

    def _getCurrentMarketValue(self) -> None:
        """ This will compute the current market value by computing the market charges when selling"""

        # PSE Charges
        #   Gross Transaction Amount    : share x market price
        #   Broker's Commission         : 0.25% x GROSS_TRANSACTION_AMOUNT
        #   Broker's Commission VAT     : 12% x BROKER_COMMISION
        #   SCCP Fee                    : 0.01% x GROSS_TRANSACTION_AMOUNT
        #   PSE Transaction Fee         : 0.005% x GROSS_TRANSACTION_AMOUNT
        #   Sales Transaction Tax	    : 0.6% of the gross trade value
        gross_transaction_amount = self.market_price * self.current_shares
        brokers_commision = 0.0025 * gross_transaction_amount
        brokers_commision = 20 if brokers_commision <= 20 else brokers_commision

        brokers_commision_vat = 0.12 * brokers_commision
        sccp_fee = 0.0001 * gross_transaction_amount
        pse_transaction_fee = 0.00005 * gross_transaction_amount
        sales_tax = 0.006 * gross_transaction_amount

        self.market_value = gross_transaction_amount - (
            brokers_commision
            + brokers_commision_vat
            + sccp_fee
            + pse_transaction_fee
            + sales_tax
        )

        # Compute the difference of the entry market price than average price
        self.gain_loss = self.market_value - (self.average_value)
        self.gain_loss_pct = self.gain_loss / self.average_value
        # Change when use multiple stocks to trade
        self.port_gain_loss = self.gain_loss
        self.port_gain_loss_pct = self.port_gain_loss / self.current_equity

    def _sell(self):
        """
        Executing the trade.
        """

        if self.gain_loss > 0:
            self.win_count += 1
            self.cumm_gain += self.gain_loss
        else:
            self.cumm_loss += abs(self.gain_loss)

        if self.isBacktest:
            # Record the trade made
            self.logs.append(
                self.current_date,
                self.current_equity,
                self.stock_name,
                self.market_price,
                self.average_price,
                self.current_shares,
                self.market_value,
                self.gain_loss,
                self.gain_loss_pct,
                "SELL",
                f"{self.gain_loss_pct % 100}%",
            )

        # Reset
        self.cumm_profit += self.gain_loss
        self.current_equity += self.market_value
        self.port_gain_loss = 0
        self.port_gain_loss_pct = 0
        self.gain_loss = 0
        self.gain_loss_pct = 0
        self.average_price = 0
        self.average_value = 0
        self.market_price = 0
        self.market_value = 0
        self.total_trade += 1
        self.position_days = 0


    def _getBoardLot(current_price):
        lot_size = 5
        current_price = float(current_price)
        # Price between 0.0001 and 0.0099
        if current_price <= 0.0099:
            lot_size = 1000000
        # Price between 0.01 and 0.049
        elif current_price <= 0.495:
            lot_size = 100000
        # Price between 0.05 and 0.495
        elif current_price <= 0.495:
            lot_size = 10000
        # Price between 0.5 and 4.99
        elif current_price <= 4.99:
            lot_size = 1000
        # Price between 5 and 49.95
        elif current_price <= 49.95:
            lot_size = 100
        # Price between 50 and 999.5
        elif current_price <= 999.5:
            lot_size = 10

        return lot_size
