# pynance

`pynance` is a Python library that provides a collection of financial formulas and calculations for quants, traders, analysts, and students. It offers functions for time value of money, bond pricing, stock valuation, option pricing, risk metrics, and more.

## Features

- **Time Value of Money (TVM) Functions**: Present Value, Future Value, Net Present Value (NPV), Internal Rate of Return (IRR).
- **Bond Pricing and Yield Calculations**: Bond Price, Yield to Maturity (YTM).
- **Stock Valuation**: Dividend Discount Model (DDM).
- **Option Pricing**: Black-Scholes Model for option pricing.
- **Risk and Performance Metrics**: Sharpe Ratio, Value at Risk (VaR).

## Installation

You can install `pynance` via pip:

```bash
pip install pynance
```


## Functions Available

- #### Time Value of Money:
  ```python
  present_value(future_value, rate, periods)
  ```
  ```python
  future_value(present_value, rate, periods)
  ```
  ```python
  npv(rate, cash_flows)
  ```
  ```python
  internal_rate_of_return(cash_flows)
  ```

- #### Bond Pricing and Yield Calculations:
  ```python
  bond_price(face_value, coupon_rate, periods, discount_rate)
  ```
  ```python
  yield_to_maturity(face_value, coupon_rate, periods, price)
  ```

- #### Stock Valuation:
  ```python
  dividend_discount_model(dividend, growth_rate, discount_rate)
  ```

- #### Option Pricing:
  ```python
  black_scholes_call(S, K, T, r, sigma)
  ```

- #### Risk and Performance Metrics:
  ```python
  sharpe_ratio(returns, risk_free_rate)
  ```
  ```python
  value_at_risk(returns, confidence_level)
  ```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -am 'Add new feature').
- Push to the branch (git push origin feature-branch). 
- Create a new Pull Request.
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or issues, please contact reddygari.rithvik@gmail.com

## Acknowledgments
This library was inspired by my interest in finance and my coursework! Thanks to all who help in maintaining and adding new features!