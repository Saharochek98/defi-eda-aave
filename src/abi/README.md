# Aave V3 Pool event ABI

`aave_v3_pool_events.json` is the minimal ABI subset for `Supply`, `Borrow`,
and `Repay`, transcribed programmatically from the official Aave interface:

https://github.com/aave/aave-v3-core/blob/master/contracts/interfaces/IPool.sol

The Solidity enum `DataTypes.InterestRateMode` is encoded as `uint8` in the ABI.
The original source snapshot is saved in `data/raw/onchain/references/IPool.sol`.
Pool address provenance:

https://github.com/bgd-labs/aave-address-book/blob/main/src/AaveV3Ethereum.sol

Pool: `0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2`, Ethereum mainnet only.
This ABI subset is not intended for sending transactions or decoding all Pool events.
