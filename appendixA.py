import pandas as pd
import numpy as np
import scipy.constants as const
from itertools import product

### CALCULATION FUNCTIONS
# calc_df()      -- calculates principal  dataframe parameters 
# normalize_df() -- normalize the columns of the DF
# r_max()        -- calculates reactive field drop-off distance for a given freq

def calc_df(df: pd.DataFrame, eps: float = eps, mu: float = mu) -> pd.DataFrame:
    # copy DF -- reset if already filled
    cols = ['a','b','R', 'N','rN']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy(deep=True)

    df['ln_b_a']    = np.log(df['b'] / df['a'])
    df['b_div_a']   = df['b'] / df['a']

    # sep approximations + get/set constraint masks
    df[['wire_prop_legal','approx_L','approx_C']] = True
    df.loc[df.b_div_a <= 2, 'wire_prop_legal'] = False
    approx_L_true_IDX = df.loc[(df.b_div_a > 90)].index
    df.loc[approx_L_true_IDX, 'approx_L'] = False
    approx_C_true_IDX = df.loc[(df.ln_b_a > 1)].index
    df.loc[approx_C_true_IDX, 'approx_C'] = False

    # Single Turn Loops
    if ('N' not in df.columns) or df['N'].max() == 1:
        # calc L and C
        df['L_approx']  = mu * df.b * np.log(df.b / df.a)
        df['L']         = mu * df.b * ( np.log((8 * df.b) / df.a) - 2 )
        df['C_approx']  = ( 2 * eps * df.b) / np.log(df.b / df.a)
        df['C']         = ( 2 * eps * df.b) / ( np.log((8 * df.b) / df.a) - 2 )

        # calc dependent params based on L & C approx rules (via index mask)...
        for (bool_approx_L, bool_approx_C) in product([True, False], repeat=2):
            L = df['L'] if (bool_approx_L == False) else df['L_approx']
            C = df['C'] if (bool_approx_C == False) else df['C_approx']
            idx_mask = df.loc[(df.approx_L == bool_approx_L
                ) & (df.approx_C == bool_approx_C)].index
            df.loc[idx_mask,'Q_approx'] = (C * df.R) / np.sqrt(L * C)
            df.loc[idx_mask,'f0_approx'] = np.power(2*np.pi*np.sqrt(L*C),-1)

        # Find volume of CI used (min. estimate), mm^3
        df['Vci_mm3'] = (4 * const.pi**2 * df.a**2  * df.b) * 1e9

    # TODO: FOR MULTI-TURN LOOPS
    elif 'N' in df.columns:

        df['L'] = mu * df.b * ( np.log((8 * df.b) / df.a) - 2 )
        def calc_self_capacitance(N, a, b, rN, mu):
            ... # requires EM Sim
        df['C'] = df.apply(lambda r: calc_self_capacitance(r.N, r.a, r.b, r.rN, mu), axis=1)

    # TURN-AGNOSTIC PARAMS
    # compute Q & f0
    df['Q']  = (2*df.R*df.b*eps)/((np.log((8*df.b)/df.a) - 2) * np.sqrt(2*np.power(df.b,2)*eps*mu))
    df['f0'] = 1 / (2 * np.pi * np.sqrt(df.L * df.C))

    # Compute initial low- and high-cutoff frequencies (3dB) ++ bandwidth
    # Fix when fL and fH are swapped... (100k / 2.4M followed reverse trend)
    fL = df['f0'] / df['Q']
    fH = df['f0'] * df['Q']
    fL_lt_fH = fL <= fH
    df['fL'], df['fH'] = np.where(fL_lt_fH, fL, fH), np.where(fL_lt_fH, fH, fL)
    df['bandwidth'] = (df['fH'] - df['fL']).abs()

    # cleanup + return
    df = df.drop(columns=['approx_L','approx_C','wire_prop_legal'])
    return df

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all columns of a dataframe safely & appended as norm_COL"""
    df = df.convert_dtypes(dtype_backend='numpy_nullable')
    # Append normalized columns to dataframe
    df = df.join((df - df.min()) / (df.max() - df.min()), rsuffix='_norm')
    df = df.drop(columns=['abin_norm','bbin_norm','z_order_idx_norm'])
    df = df.convert_dtypes(dtype_backend='pyarrow')
    return df

def r_max(freq) -> pd.Series|pd.DataFrame:
    """Compute Max Detectable Distance of Near Field at a given Frequency"""
    if isinstance(freq, Iterable):
        return pd.DataFrame([r_max(f) for f in freq])
    else:
        s = pd.Series({'freq': freq, 'wav': 0, 'r': 0})
        s.wav = const.nu2lambda(s.freq)
        s.r = np.sqrt(s.wav/(2*const.pi))
        return s

### HELPER FUNCS
# gen_data()    -- helper to generate clean dataframe structure with calc_df()
# add_z_order() -- post-processing. compute z-order curve to reduce a-b
#                  dimensionality and increase interpretability.

def gen_data(
    eps: float,
    mu: float,
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    R: int = 50,
    N: Tuple[int, int] | None = None,
    rN: Tuple[float, float, float] | None = None,
    filter_f0: Tuple[float, float] | None = None,
    filter_band_inc_freq: float = None
    ) -> pd.DataFrame:
    """ Accept input ranges & Constants, return DataFrame of all calculated values

    Args:
        eps (float):  Permittivity of medium (e.g. vacuum, Ag)
        mu (float):  Permeability of medium (e.g. vacuum, Ag)
        R (int):  Load resistance
        a (Tuple[float, float, float] | None):  Wire radius
        b (Tuple[float, float, float] | None):  Loop radius
        N (Tuple[int, int] | None):  No. of turns in small loop
        rN (Tuple[float, float, float] | None):  Separation between loops if there are turns.
        filter_f0 (Tuple[float, float] | None):  Filter output by resonant frequency ranges (in Hz)
        filter_band_inc_freq (float):  Filter output by frequency in bandwidth

    Returns:
        pd.DataFrame: calculated dataframe within params
    """
    a = None if a is None else np.arange(start=a[0], stop=a[1], step=a[2])
    b = None if b is None else np.arange(start=b[0], stop=b[1], step=b[2])
    N = [1] if N is None else np.arange(start=N[0], stop=N[1], step=1)
    rN = [0] if rN is None else np.arange(start=rN[0], stop=rN[1], step=rN[2])
    R = list(R) if isinstance(R,Iterable) else [R]

    dims = {'a':a, 'b':b, 'R':R, 'N':N, 'rN':rN }
    dims = { k:v for k,v in dims.items() if v is not None }

    mindex = pd.MultiIndex.from_product(dims.values(), names=dims.keys())
    df = pd.DataFrame(index=mindex).reset_index()

    df = calc_df(df, eps=eps, mu=mu) #.convert_dtypes(dtype_backend='pyarrow')

    if filter_f0 is not None:
        low_freq, high_freq = min(filter_f0), max(filter_f0)
        print(f"Filtering dataframe by resonant frequency range: [{low_freq/1e6}, {high_freq/1e6}] MHz")
        df = df.loc[(df.f0 >= low_freq) & (df.f0 <= high_freq)]

    if filter_band_inc_freq is not None:
        print(f"Filtering dataframe by resonant frequency range: {filter_band_inc_freq/1e6} MHz")
        df = df.loc[(df.fL <= filter_band_inc_freq) & (df.fH >= filter_band_inc_freq)]

    return df

def add_z_order(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """Add a Z-order index to the dataframe based on two columns"""
    df = df.copy(deep=True)
    def morton_encode(x, y):
        """ Computes the Morton encoding (Z-order) for two integers x and y. """
        try:
            z, max_size = 0, max(x.bit_length(), y.bit_length())
        except Exception as e:
            return pd.NA
        print(f"x, y, z, max_size = {x}, {y}, {z}, {max_size}")
        for i in range(max_size):
            z |= (x & 1 << i) << i | (y & 1 << i) << (i + 1)
            #z |= (x & 1 << i) << (2 * i) | (y & 1 << i) << (2 * i + 1)
        return z

    df[col_a+'bin'] = pd.cut(df[col_a], bins=1000, ordered=True).cat.codes
    df[col_b+'bin'] = pd.cut(df[col_b], bins=1000, ordered=True).cat.codes
    df['z_order_idx'] = df.apply(lambda r: morton_encode(r[col_a+'bin'], r[col_b+'bin']), axis=1)
    return df
