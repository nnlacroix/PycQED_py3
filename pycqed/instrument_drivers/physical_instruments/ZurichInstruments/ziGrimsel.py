'''
Created on 30.09.2019
File contains all necessary high-level functions for configuring and controlling a Grimsel device

For all channel variables: 0 represents the leftmost connector on the MBRD and 7 the rightmost! This is in contradiction
to the data converter numbers, which are enumerated the other way. Thus, the channels are intentionally swapped
in each corresponding function!
@author: fabp
'''

import time
import numpy as np
import zhinst.ziPython as ziapi
#from scipy.optimize import leastsq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#from IPython.display import clear_output

# Load the LabOne API
import os
import struct
import sys
import math
import csv
import pickle

#from backend_stats import colors

# -----------------------------------------
# Definitions for MBRD PS GPOs
# -----------------------------------------
GPO_SPI_RSTN = 0
GPO_I2C_RSTN = 1
GPO_FANFULL = 2
GPO_EXTCLK_SEL = 3
GPO_EXTREF_SEL = 4
GPO_FE_REFCLK_EN = 5
GPO_LED = 6
GPO_DAC_AVTT_SET = 7
GPO_MEZZ_PWREN = 8
GPO_SYNTH_PLL_SYNC = 9
GPO_FPGACLK_RSTN = 10

GPO_def = [GPO_SPI_RSTN, GPO_I2C_RSTN, GPO_FANFULL, GPO_EXTCLK_SEL, GPO_EXTREF_SEL, GPO_FE_REFCLK_EN, GPO_LED, GPO_DAC_AVTT_SET, GPO_MEZZ_PWREN, GPO_SYNTH_PLL_SYNC, GPO_FPGACLK_RSTN]

GPI_REFCLK_LOCK_DET = 0
GPI_SPOW_SHUTDOWN = 1
GPI_REFCLK_DET = 2

# -----------------------------------------
# Settings for MBRD PS GPOs
# -----------------------------------------
SPI_RSTN = 1            # 0 = reset, 1 = normal operation
I2C_RSTN = 1            # 0 = reset, 1 = normal operation
FANFULL = 0             # 0 = normal operation, 1 = all Fans on Full speed
EXTCLK_SEL = 1          # 0 = ZSYNC, 1 = ExtRef
EXTREF_SEL = 0          # 0 = DAC (trim), 1 = External
FE_REFCLK_EN = 0        # 0 = normal operation, 1 = 100 MHz Ref enabled
LED = 0                 # 0 = Off, 1 = On
DAC_AVTT_SET = 0        # 0 = 2.5V, 1 = 3.0V
MEZZ_PWREN = 0          # 0 = On, 1 = Off
SYNTH_PLL_SYNC = 0      # 0 = Off, 1 = Sync pulse
FPGACLK_RSTN = 1        # 0 = reset, 1 = normal operation

GPO_set = [SPI_RSTN, I2C_RSTN, FANFULL, EXTCLK_SEL, EXTREF_SEL, FE_REFCLK_EN, LED, DAC_AVTT_SET, MEZZ_PWREN, SYNTH_PLL_SYNC, FPGACLK_RSTN]

# -----------------------------------------
# Settings specifically for the Upconverter
# -----------------------------------------
ATT_VAL_uc = 0x00       # 0x00=0dB, 0x01=0.5dB, 0x3F=31.5dB
OUT_SW_1_1 = 0x1        # 0 = Off 1 = RF OUT
OUT_SW_1_2 = 0x0        # 0 = Off 1 = RF Tap

OUT_SW_2_1 = 0x1        # 0 = Off 1 = BB
OUT_SW_2_2 = 0x0        # 0 = Off 1 = RF
FILTER_BANK = 0x01      # 0 = 1.5G 1 = 3G 2 = 3-6G 4 = 5-10G
RF_PA_EN = 0x1          # 0 = On 1 = Off
RF_BB_IN_uc = 0x1       # 0 = RF 1 = BB
AUX_IN_ATT = 0x0        # 0 = Off 1 = On
CAL_ADC_uc = 0x0        # 0 = Off 1 = On

BB_POST_ATT = 0x0       # 0 = Off 1 = On
BB_LMH_PD_uc = 0x0      # 0 = LMH On 1 = LMH Off
RF_ATT = 0x1            # 0 = On 1 = Off
TRG_IN_ATT_uc = 0x0     # 0 = 1k 1 = 50R
AUX_LMH_PD_uc = 0x1     # 0 = LMH On 1 = LMH Off
AUX_IN_SEL = 0x0        # 0 = AUX 1 = RF Tap
CAL_REF_uc = 0x0        # 0 = Off 1 = On
CAL_DAC_uc = 0x0        # 0 = Off 1 = On

TCA6424_upconv_out0 = 0x0
TCA6424_upconv_out1 = 0x0
TCA6424_upconv_out2 = 0x0

# -----------------------------------------
# Settings specifically for the Downconverter
# -----------------------------------------
ATT_VAL_dc = 0x04       # 0x00=0dB, 0x01=0.5dB, 0x3F=31.5dB
RF_PREAMP = 0x0         # 0 = Off, 1 = On
RF_POSTAMP = 0x0        # 0 = Off, 1 = On
TRG_IN_ATT_dc = 0x0     # 0 = 1k, 1 = 50R
CAL_DAC_dc = 0x0        # 0 = Off 1 = On
CAL_ADC_dc = 0x0        # 0 = Off 1 = On
CAL_REF_dc = 0x0        # 0 = Off 1 = On
RF_BB_IN_dc = 0x0       # 0 = RF 1 = BB
BB_BW = 0x0             # 0 = Full, 1 = 2 GHz
BB_LMH_PD_dc = 0x1      # 0 = LMH On 1 = LMH Off
BB_Att = 0x0            # 0 = Off, 1 = On
BIST_SW = 0b00000001    # 0 = Off, 0b00000001 = RF Input, 0b01000000 = DAC_BIST
IN_SW = 0b00000010      # 0 = Off, 0b00000010 = RF Path,  0b00100000 = BB Path

TCA6424_downconv_out0 = 0x0
TCA6424_downconv_out1 = 0x0
TCA6424_downconv_out2 = 0x0

# -----------------------------------------
# Settings for the TCA6424A GPIO Chip
# -----------------------------------------
TCA6424_addr = 0x22
TCA6424_reg_iodir = 0x8C
TCA6424_reg_out = 0x84
TCA6424_reg_in = 0x80
TCA6424_iodir = 0x000000

# -----------------------------------------
# Settings for the MCP23017 GPIO Chip
# -----------------------------------------
MCP23017_addr_frontend = 0x20
MCP23017_reg_gpioa = 0x12
MCP23017_addr_FP = 0x26
MCP23017_addr_mbrd = 0x27

# -----------------------------------------
# Registers for the MAX31790
# -----------------------------------------
MAX31790_mux = 10
MAX31790_addr = 0x2F
MAX31790_reg_config_fan1 = 0x02
MAX31790_reg_config_fan2 = 0x03
MAX31790_reg_config_fan3 = 0x04
MAX31790_reg_config_fan4 = 0x05
MAX31790_reg_config_fan5 = 0x06
MAX31790_reg_config_fan6 = 0x07
MAX31790_reg_PWM1_target = 0x40
MAX31790_reg_PWM3_target = 0x44
MAX31790_reg_PWM1_current = 0x30
MAX31790_reg_tach1 = 0x18
MAX31790_reg_tach2 = 0x1A
MAX31790_reg_tach3 = 0x1C
MAX31790_reg_tach4 = 0x1E
MAX31790_reg_tach5 = 0x20
MAX31790_reg_tach6 = 0x22

fan_mode = 0x0              # 0 = PWM, 1 = RPM
fan_spinup = 0x0            # 0 = no spinup, 1 = 2xTach|0.5s, 2 = 2xTach|1s, 3 = 2xTach|2s
fan_monitor = 0x0           # 0 = Control, 1 = Monitor
fan_tach_en = 0x1           # 0 = Disable, 1 = Enable
fan_tach_locked = 0x0       # 0 = TACH, 1 = Locked
fan_locked_pol = 0x0        # 0 = Low, 1 = high
fan_pwm_tach = 0x0           # 0 = PWM, 1 = TACH

MAX31790_PWM = 255
MAX31790_fan_PSU = (fan_mode<<7)|(fan_spinup<<6)|(fan_monitor<<4)|(fan_tach_en<<3)|(fan_tach_locked<<2)|(fan_locked_pol<<1)|(fan_pwm_tach<<0)
MAX31790_fan_CASE = (fan_mode<<7)|(fan_spinup<<6)|(fan_monitor<<4)|(fan_tach_en<<3)|(fan_tach_locked<<2)|(fan_locked_pol<<1)|(fan_pwm_tach<<0)
MAX31790_fan_FPGA = (fan_mode<<7)|(fan_spinup<<6)|(fan_monitor<<4)|(fan_tach_en<<3)|(fan_tach_locked<<2)|(fan_locked_pol<<1)|(fan_pwm_tach<<0)

# -----------------------------------------
# Registers for the LMH6401
# -----------------------------------------
LMH6401_addr_revID = 0
LMH6401_addr_prodID = 1
LMH6401_addr_gain = 2
LMH6401_addr_res = 3
LMH6401_addr_thGain = 4
LMH6401_addr_thFreq = 5

LMH6401_att =  20        # Attenuation value in dB. 0x00 = 0dB att = 26dB gain, 0x1A = 0dB gain, 0x20-0x3F = -6 dB gain
LMH6401_pd = 0
LMH6401_reg_gain = (0<<7)|(LMH6401_pd<<6)|(LMH6401_att<<0)

# -----------------------------------------
# Definitions for AD9508 devices
# -----------------------------------------
RefOut_mux = 10
AD9508_addr = 0x6F
AD9508_reg_ID = 0x0C
AD9508_reg_out0drvmode = 0x19
AD9508_reg_out0div = 0x15
AD9508_reg_out0div1 = 0x16
AD9508_reg_in = 0x80
AD9508_iodir = 0x000000

AD9508_reg_out1drvmode = 0x1F
AD9508_reg_out2drvmode = 0x25
AD9508_reg_out3drvmode = 0x2B

# -----------------------------------------
# Definitions for MBRD PS GPOs
# -----------------------------------------
GPO_SPI_RSTN = 0
GPO_I2C_RSTN = 1
GPO_FANFULL = 2
GPO_EXTCLK_SEL = 3
GPO_EXTREF_SEL = 4
GPO_FE_REFCLK_EN = 5
GPO_LED = 6
GPO_DAC_AVTT_SET = 7
GPO_MEZZ_PWREN = 8
GPO_SYNTH_PLL_SYNC = 9
GPO_FPGACLK_RSTN = 10

GPO_def = [GPO_SPI_RSTN, GPO_I2C_RSTN, GPO_FANFULL, GPO_EXTCLK_SEL, GPO_EXTREF_SEL, GPO_FE_REFCLK_EN, GPO_LED, GPO_DAC_AVTT_SET, GPO_MEZZ_PWREN, GPO_SYNTH_PLL_SYNC, GPO_FPGACLK_RSTN]

GPI_REFCLK_LOCK_DET = 0
GPI_SPOW_SHUTDOWN = 1
GPI_REFCLK_DET = 2

# global LMX2595 PLL definitions
verification_register = 36
check_register = 106
long_prog = 1
reg_start_short = 78
reg_start_long = 112

# global variables used for PLL programming
reg_array = []
reg = []
current_line = 0
PLL_Power = 31



# Debug register assignments
DEBUG_REG_REG_MIXER_FREQUENCIES_OFFSET32  = 0x00
DEBUG_REG_REG_IQ_VALUES_OFFSET32          = 0x08
# DEBUG_REG_REG_OUTPUT_EN_OFFSET32 is deprecated. Use /RAW/DACSOURCES/<n>/PATHS/<m>/ON instead.
# DEBUG_REG_REG_OUTPUT_EN_OFFSET32          = 0x0E
DEBUG_REG_REG_OUTPUT_CURRENT_OFFSET32     = 0x0F
DEBUG_REG_REG_DAC_MIXER_TEST_ID           = 0x15
DEBUG_REG_REG_OUTPUT_SINES_GAIN           = 0x16
# DEBUG_REG_REG_OUTPUT_RANDOM_EN is deprecated. Use /RAW/DACSOURCES/<n>/SOURCESELECT instead.
# DEBUG_REG_REG_OUTPUT_RANDOM_EN            = 0x17
DEBUG_REG_REG_OUTPUT_ZERO_Q               = 0x18
DEBUG_REG_REG_TRIG_CTRL_START             = 0x26
DEBUG_REG_REG_TRIG_CTRL_SRC_SEL           = 0x27


colors = [0, 0, 0, 0]

# -----------------------------------------
# Settings for the TCA6424A GPIO Chip
# -----------------------------------------
INA226_mux = 9
INA226_addr = [0x40, 0x41, 0x42, 0x43, 0x44]
INA226_dev_names = ["+12V Input","ADC_AVCC","DAC_AVCC","ADC_AVCCAUX"]
INA226_PSU1_addr = 0x40
INA226_PSU2_addr = 0x41
INA226_ADCAVCC_addr = 0x42
INA226_DACAVCC_addr = 0x43
INA226_ADCAVCCAUX_addr = 0x44

INA226_reg_config = 0x00
INA226_reg_shunt = 0x01
INA226_reg_bus = 0x02
INA226_reg_pow = 0x03
INA226_reg_cur = 0x04
INA226_reg_cal = 0x05
INA226_reg_ID = 0xFE

INA226_PSU_cal = 10240
INA226_dataconv_cal = 10240

INA226_config =  0x927
INA226_cur_lsb = [0.001, 0.001, 0.0001, 0.0001, 0.0001]
INA226_cal = [INA226_PSU_cal, INA226_PSU_cal, INA226_dataconv_cal, INA226_dataconv_cal, INA226_dataconv_cal]

# -----------------------------------------
# Settings for the TCA6424A GPIO Chip
# -----------------------------------------
LTC2991_addr_1 = 0x48
LTC2991_addr_2 = 0x49
LTC2991_addr_2_alt = 0x4A

LTC2991_reg_status_lo = 0x00
LTC2991_reg_status_hi = 0x01
LTC2991_reg_v1_v4_ctrl = 0x06
LTC2991_reg_v5_v8_ctrl = 0x07
LTC2991_reg_vcc_tint = 0x08
LTC2991_reg_v1 = 0x0A
LTC2991_reg_v2 = 0x0C
LTC2991_reg_v3 = 0x0E
LTC2991_reg_v4 = 0x10
LTC2991_reg_v5 = 0x12
LTC2991_reg_v6 = 0x14
LTC2991_reg_v7 = 0x16
LTC2991_reg_v8 = 0x18
LTC2991_reg_VCC = 0x1C

LTC2991_status_hi = 0xF8
LTC2991_v1_v4_ctrl = 0x00
LTC2991_v5_v8_ctrl = 0x00
LTC2991_vcc_tint = 0x10
LTC2991_SINGLE_ENDED_lsb = 0.000305176



# # -----------------------------------------
# # Definitions for Scopes
# # -----------------------------------------
# Grimsel.record_length = 2**18
# N = Grimsel.record_length/2
# window_fct_name = "hanning"
# peak_threshold = -80
# Fs = 4e9  # [Hz]
# # n_fft = Grimsel.record_length
# #f = np.linspace(0, Fs/2, N, endpoint=True)
# t = np.array(range(0, Grimsel.record_length)) / Fs
# Grimsel.channel_enable = [0, 0, 0, 0]
# Grimsel.input_select = [0, 0, 0, 0]
# Grimsel.recorded_data = [[],[],[],[]]
# #peaks = {}

# define format for plotting
plot_color = ['C0','C1','C2','C3']
plot_legent_txt = []

# -----------------------------------------
# Definitions for LTC2655
# -----------------------------------------
LTC2655_mux = 13
LTC2655_addr = 0x10
LTC2655_reg_update_n = 0x3
LTC2655_reg_DAC_A = 0x0
LTC2655_reg_DAC_B = 0x1
LTC2655_reg_DAC_C = 0x2
LTC2655_reg_DAC_D = 0x3
LTC2655_reg_DAC_E = 0x4
LTC2655_reg_DAC_F = 0x5
LTC2655_reg_DAC_G = 0x6
LTC2655_reg_DAC_H = 0x7
LTC2655_reg_DAC_ALL = 0xF
LTC2655_lsb = 62.501E-06
LTC2655_value = 0.000/LTC2655_lsb
LTC2655_increment = 20e-3/LTC2655_lsb    #mV

PMBus_mux = 8

class GrimselDEV():
    def __init__(self,a_daq='',a_devid=''):
        self.daq = a_daq
        self.devid = a_devid
        self.INA226_cur_array=[0,0,0,0]
        self.INA226_volt_array=[0,0,0,0]
        self.INA226_pow_array=[0,0,0,0]
        self.INA226_addr = INA226_addr
        self.INA226_cal = INA226_cal
        self.INA226_cur_lsb = INA226_cur_lsb
        # -----------------------------------------
        # Definitions for Scopes
        # -----------------------------------------
        self.record_length = 2 ** 18
        self.N = self.record_length / 2
        self.window_fct_name = "hanning"
        self.Fs = 4e9  # [Hz]
        # n_fft = Grimsel.record_length
        # f = np.linspace(0, Fs/2, N, endpoint=True)
        self.t = np.array(range(0, self.record_length)) / self.Fs
        self.channel_enable = [0, 0, 0, 0]
        self.input_select = [0, 0, 0, 0]
        self.recorded_data = [[], [], [], []]
        self.input_signal_psd_dB = [[],[],[],[]]
        self.peaks = {}
        self.peak_threshold = -80

        # definitions for nyquist and freeze settings
        self.freeze = np.ones(8)
        self.zones = np.zeros(8)
        self.nyquist_zone_setting = 0
        self.fig_num = 0
        self.n_fft = self.record_length




# ----------------------------
# Function definitions
# ----------------------------

def transdaq(daq, device):  # added by FH in order to assure compatibilty to ZIBackend
    global Grimsel
    Grimsel=GrimselDEV(daq,device)

def grmConnect(port=8004, host='localhost', apilevel=6, device_name='dev12004'):
    global Grimsel
    Grimsel =GrimselDEV(ziapi.ziDAQServer(host, port, apilevel),device_name)
    Grimsel.daq.connectDevice(Grimsel.devid, '1gbe')
    return

def grmDisconnect(device_name):
    # define device name
    Grimsel.daq.disconnectDevice(device_name)
    return

# Disable Printout
def disablePrint():
    # sys.stdout = open(os.devnull, 'w')
    return

# Restore printout
def enablePrint():
    # sys.stdout = sys.__stdout__
    return

def getGPI(input=0):
    #REFCLK_LOCK_DET        0
    #SPOW_SHUTDOWN          1
    #RD_REFCLK_DET          2

    temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/GPIS/{input}/VALUE')
    return temp

def setGPO(node=6, value=0):
    #PS_SPI_RSTN            0
    #PS_I2C_RSTN            1
    #PS_FORCE_FAN_FULL      2
    #PS_EXTCLK_SOURCE_SEL   3    0=ZSync, 1=Extref
    #PS_EXTCLK_SEL          4    0=Trimming-DAC, 1=EXTREF
    #PS_FE_REFCLK_EN        5
    #LED1                   6
    #PS_DAC_AVTT_SET        7
    #MEZZ_PWREN             8
    #PS_SYNTH_PLL_SYNC      9
    #PS_FPGACLK_RSTN       10

    Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/GPOS/{node}/VALUE', value)
    return

def setGPO_All(spi_rst=1, i2c_rst=1, fanfull=0, extclk=1, extref=0, fe_refclk_en=0, led=0, dacavtt_set=0, mezz_pwren=0, synth_pll_sync=0, fpgaclk_rst=1):
    SPI_RSTN = spi_rst              # 0 = reset, 1 = normal operation
    I2C_RSTN = i2c_rst              # 0 = reset, 1 = normal operation
    FANFULL = fanfull               # 0 = normal operation, 1 = all Fans on Full speed
    EXTCLK_SEL = extclk             # 0 = ZSYNC, 1 = ExtRef
    EXTREF_SEL = extref             # 0 = DAC (trim), 1 = External
    FE_REFCLK_EN = fe_refclk_en     # 0 = normal operation, 1 = 100 MHz Ref enabled
    LED = led                       # 0 = Off, 1 = On
    DAC_AVTT_SET = dacavtt_set      # 0 = 2.5V, 1 = 3.0V
    MEZZ_PWREN = mezz_pwren         # 0 = On, 1 = Off
    SYNTH_PLL_SYNC = synth_pll_sync # 0 = Off, 1 = Sync pulse
    FPGACLK_RSTN = fpgaclk_rst      # 0 = reset, 1 = normal operation

    GPO_set = [SPI_RSTN, I2C_RSTN, FANFULL, EXTCLK_SEL, EXTREF_SEL, FE_REFCLK_EN, LED, DAC_AVTT_SET, MEZZ_PWREN, SYNTH_PLL_SYNC, FPGACLK_RSTN]

    for x in range(len(GPO_def)):
        setGPO(GPO_def[x], GPO_set[x])
    return

def reverseBits(num,bitSize):
    # convert number into binary representation
    # output will be like bin(10) = '0b10101'
    binary = bin(num)

    # skip first two characters of binary
    # representation string and reverse
    # remaining string and then append zeros
    # after it. binary[-1:1:-1]  --> start
    # from last character and reverse it until
    # second last character from left
    reverse = binary[-1:1:-1]
    reverse = reverse + (bitSize - len(reverse))*'0'

    # converts reversed binary string into integer
    return int(reverse,2)

def power2att_upconv(power=0.0, freq=1.0):
    power_dict = {
    0.0: 10.0,
    1.0: 8.1,
    1.1: 7.6,
    1.2: 7.3,
    1.3: 7.0,
    1.4: 6.6,
    1.5: 6.3,
    1.6: 5.4,
    1.7: 6.2,
    1.8: 5.8,
    1.9: 5.5,
    2.0: 5.5,
    2.1: 5.7,
    2.2: 5.8,
    2.3: 5.7,
    2.4: 5.6,
    2.5: 5.9,
    2.6: 5.7,
    2.7: 5.5,
    2.8: 5.2,
    2.9: 4.8,
    3.0: 4.3,
    3.2: 3.7,
    3.4: 2.9,
    3.6: 3.1,
    3.8: 3.4,
    4.0: 2.9,
    4.2: 2.2,
    4.4: 2.4,
    4.6: 1.7,
    4.8: 1.6,
    5.0: 1.1,
    5.2: 0.5,
    5.4: 0.1,
    5.6: 0.8,
    5.8: -0.1,
    6.0: -0.8,
    6.2: -0.7,
    6.4: -0.9,
    6.6: -0.9,
    6.8: -0.8,
    7.0: -1.1,
    7.2: -1.6,
    7.4: -2.3,
    7.6: -3.0,
    7.8: -3.7,
    8.0: -4.6,
    8.2: -4.6,
    8.4: -4.6,
    8.6: -4.6,
    8.8: -4.6,
    9.0: -4.6}

    # calculate attenuation value from calibration file and requested output power
    Upconv_ref_lvl = power_dict[0.0]
    att = 2*Upconv_ref_lvl+power_dict.get(freq)-power

    # limit minimum attenuation value to 0. Maximum value is checked in setUpconverter(), so no need to do this here as well
    if att >= 0:
        return att
    else:
        return 0

def power2dacpwr(argument):
    power = {
        1.6: 0,
        1.7: 0,
        1.8: 0,
        1.9: 2,
        2.0: 0,
        2.1: 0,
        2.2: 0,
        2.3: 0,
        2.4: 0,
        2.5: 0,
        2.6: 0,
        2.7: 0,
        2.8: 0,
        2.9: 0,
        3.0: 0,
        3.2: 0,
        3.4: 0,
        3.6: 0,
        3.8: 0,
        4.0: 0,
        4.2: 0,
        4.4: 0,
        4.6: 0,
        4.8: 0,
        5.0: 0,
        5.2: 0,
        5.4: 0,
        5.6: 0,
        5.8: 1,
        6.0: -2,
        6.2: 1,
        6.4: -2,
        6.6: -2,
        6.8: -2,
        7.0: -4,
        7.2: -4,
        7.4: -4,
        7.6: 0,
        7.8: 0,
        8.0: 0
        }
    closestkey = min(power, key=lambda x: abs(x - argument))
    return power[closestkey]

def freq2Power(argument):
    #argument = round(argument-12,1)
    power = {
        1.5: 31,
        1.6: 8,
        1.7: 8,
        1.8: 8,
        1.9: 8,
        2.0: 10,
        2.1: 10,
        2.2: 10,
        2.3: 8,
        2.4: 6,
        2.5: 10,
        2.6: 8,
        2.7: 8,
        2.8: 8,
        2.9: 8,
        3.0: 8,
        3.2: 6,
        3.4: 4,
        3.6: 8,
        3.8: 8,
        4.0: 8,
        4.2: 7,
        4.4: 8,
        4.6: 8,
        4.8: 8,
        5.0: 12,
        5.2: 15,
        5.4: 16,
        5.6: 31,
        5.8: 31,
        6.0: 9,
        6.2: 16,
        6.4: 12,
        6.6: 31
        }
    # power = { 1.0: 31}
    closestkey = min(power, key=lambda x: abs(x - argument))
    return power[closestkey]

def PLLregs(a_freq):
    PLL={'R112': 0x700000,
         'R111': 0x6F0000,
       'R110': 0x6E0000,
       'R109': 0x6D0000,
       'R108': 0x6C0000,
       'R107': 0x6B0000,
       'R106': 0x6A0011,
       'R105': 0x690021,
       'R104': 0x680000,
       'R103': 0x670000,
       'R102': 0x660000,
       'R101': 0x650011,
       'R100': 0x640000,
       'R99': 0x630000,
       'R98': 0x620400,
       'R97': 0x610888,
       'R96': 0x600000,
       'R95': 0x5F0000,
       'R94': 0x5E0000,
       'R93': 0x5D0000,
       'R92': 0x5C0000,
       'R91': 0x5B0000,
       'R90': 0x5A0000,
       'R89': 0x590000,
       'R88': 0x580000,
       'R87': 0x570000,
       'R86': 0x560000,
       'R85': 0x55F600,
       'R84': 0x540001,
       'R83': 0x530000,
       'R82': 0x520A00,
       'R81': 0x510000,
       'R80': 0x50CCCC,
       'R79': 0x4F004C,
       'R78': 0x4E0003,
       'R77': 0x4D0000,
       'R76': 0x4C000C,
       'R75': 0x4B0800,
       'R74': 0x4A0000,
       'R73': 0x49003F,
       'R72': 0x480000,
       'R71': 0x470081,
       'R70': 0x46C350,
       'R69': 0x450000,
       'R68': 0x4403E8,
       'R67': 0x430000,
       'R66': 0x4201F4,
       'R65': 0x410000,
       'R64': 0x401388,
       'R63': 0x3F0000,
       'R62': 0x3E0322,
       'R61': 0x3D00A8,
       'R60': 0x3C0000,
       'R59': 0x3B0001,
       'R58': 0x3A8001,
       'R57': 0x390020,
       'R56': 0x380000,
       'R55': 0x370000,
       'R54': 0x360000,
       'R53': 0x350000,
       'R52': 0x340820,
       'R51': 0x330080,
       'R50': 0x320000,
       'R49': 0x314180,
       'R48': 0x300300,
       'R47': 0x2F0300,
       'R46': 0x2E07FE,
       'R45': 0x2DD0C0,
       'R44': 0x2C1F80,
       'R43': 0x2B0000,
       'R42': 0x2A0000,
       'R41': 0x290004,
       'R40': 0x280000,
       'R39': 0x270001,
       'R38': 0x260000,
       'R37': 0x250204,
       'R36': 0x240082,
       'R35': 0x230004,
       'R34': 0x220000,
       'R33': 0x211E21,
       'R32': 0x200393,
       'R31': 0x1F03EC,
       'R30': 0x1E318C,
       'R29': 0x1D318C,
       'R28': 0x1C0488,
       'R27': 0x1B0002,
       'R26': 0x1A0DB0,
       'R25': 0x190C2B,
       'R24': 0x18071A,
       'R23': 0x17007C,
       'R22': 0x160001,
       'R21': 0x150401,
       'R20': 0x14E048,
       'R19': 0x1327B7,
       'R18': 0x120064,
       'R17': 0x11012C,
       'R16': 0x100080,
       'R15': 0x0F064F,
       'R14': 0x0E1E70,
       'R13': 0x0D4000,
       'R12': 0x0C5001,
       'R11': 0x0B0018,
       'R10': 0x0A10D8,
       'R9': 0x090604,
       'R8': 0x082000,
       'R7': 0x0740B2,
       'R6': 0x06C802,
       'R5': 0x0500C8,
       'R4': 0x040A43,
       'R3': 0x030642,
       'R2': 0x020500,
       'R1': 0x010808,
       'R0': 0x002518}
    PLL['R36'] = 0x240000
    if a_freq <= 15:
        PLL['R36'] += (int(a_freq*10) << 0)
        PLL['R45'] = 0x2DC8DF
        PLL['R37'] = 0x250204
        PLL['R27'] = 0x1B0002
    else:
        PLL['R36'] += (int(a_freq*5) << 0)
        PLL['R45'] = 0x2DD0DF
        PLL['R37'] = 0x250104
        PLL['R27'] = 0x1B0003

    return [PLL[x] for x in PLL]

def setPLL(channel=0, freq=1, power=0, powerdown=1, reset=0):
    SPI_chan = 3
    #SPI_OE = 0
    PLL_Power = 0
    LMX2595_PD = powerdown
    SPI_OE = int(np.floor(channel/2))

    if power==0:
        PLL_Power = freq2Power(freq)
    else:
        PLL_Power = power

    freq += 12

# ----- old variant -----------------------------------------------------------
#     if (freq <= 15) | (freq >= 15.5):
#         file_path = "LMX2595_%dG.txt" %(np.round(freq))
#     else:
#         file_path = "LMX2595_%dG.txt" %(np.ceil(freq))
#     if LMX2595_PD==0:
#         print(" Programming PLL #%d to %f GHz " %(SPI_OE, freq))
#     else:
#         print(" Powering down PLL #%d"%(SPI_OE))
#     f = open(file_path, "r")
#
#     for line in f:
#         current_line = line
#         result = current_line.find("0x")
#         #print(x)
#         if result > 0:
#             reg_array.append(int(current_line[result+2:result+8],16))
#
#     reg = reg_array
#     reg.reverse()
#     f.close()
# -----------------------------------------------------------------------------

# ----- new variant -----------------------------------------------------------
    freqr=freq
    if LMX2595_PD==0:
        print(" Programming PLL #%d to %f GHz with power PLL_Power %d" %(SPI_OE, freqr,PLL_Power))
    else:
        print(" Powering down PLL #%d"%(SPI_OE))

    reg = PLLregs(freqr)
    reg.reverse()
    reg[44] = (reg[44] & 0xFF00FF) | (PLL_Power << 8)
    #print("Register 36 content: %d" %(reg[36]&0x00FFFF))

#    range_PLLs = list(range(4))
    # choose appropriate SPI channel
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan) # PS_SPI0_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', ~(1<<SPI_OE)) # enable channel SPIM0_OE3
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (1 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()

    #print(" Check PLL status...")
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (1<<23)|(check_register<<16)) # enable channel SPIM0_OE3
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (2 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()
    status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/STATUS')
    if (status & 0x0001):
        rd_data = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/READDATA')
        #print("Read data: 0x%06X" % (rd_data))
        if rd_data == 0xFFFF:
            print(" Device not available, exiting...")
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan) # PS_SPI0_SS0 -> SPIM0_OEx
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', 0xFF) # enable channel SPIM0_OE3
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (1 << 0)) # nothing to read, 2 bytes to write
            Grimsel.daq.sync()
            return


    if reset == 1:
        #set RESET bit in register 0
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (0<<23)| (reg[0] | 2)) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()

        time.sleep(0.1)

        # remove reset bit in register 0 and continue
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (0<<23)| (reg[0])) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()

    # check for PD request
    if LMX2595_PD:
        #print(" Powering Down PLL %d..." %(SPI_OE), end="")
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (0<<23)| (reg[0] | 1)) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()
        #print(" done")
    # do the programming
    else:
        # check whether upper registers were already programmed earlier on
        print(" Check programming status...")
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (1<<23)|(check_register<<16)) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (2 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()
        status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/STATUS')
        if (status & 0x0001):
            # data readout has no meaning here as we were not asking any, just keep it for consistency
            rd_data = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/READDATA')
            #print("Read data: 0x%06X" % (rd_data))
            if rd_data == reg[check_register]&0x00FFFF:
                long_prog = 0
                print(" Device was not reset, using short programming mode")
            else:
                long_prog = 1
                print(" Device was reset, using long programming mode")

        else:
            print(" SPI access failed: 0x%04X\n" % (status))

        # remove FCAL
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', reg[0] & ~(0x000008)) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()

        # do the register programming in reverse order
        if long_prog:
            for x in reversed(range(reg_start_long+1)):
                #print(x)
                Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
                Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (0<<23)|(reg[x])) # enable channel SPIM0_OE3
                Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
                Grimsel.daq.sync()

        else:
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (1<<23)|(verification_register<<16)) # enable channel SPIM0_OE3
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (2 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
            Grimsel.daq.sync()
            status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/STATUS')
            if (status & 0x0001):
                # data readout has no meaning here as we were not asking any, just keep it for consistency
                rd_data = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/READDATA')
                #print("Read data: 0x%06X" % (rd_data))
                if rd_data != reg[verification_register]&0x00FFFF:
                    for x in reversed(range(reg_start_short+1)):
                        #print(x)
                        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
                        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (0<<23)|(reg[x])) # enable channel SPIM0_OE3
                        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
                        Grimsel.daq.sync()
                else:
                    print(" PLL already programmed...")
                # re-write reg44 to remove PD
                Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
                Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (0<<23)|(reg[44])) # enable channel SPIM0_OE3
                Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
                Grimsel.daq.sync()

        time.sleep(0.1)

        # remove FCAL once again
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', reg[0] & ~(0x000008)) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()

        time.sleep(0.1)

        # set FCAL to calibrate PLL & VCO to programmed Frequency
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (0<<23)| (reg[0] | 0x000008)) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()

        print(" Programming done")

        # read back N divider register to verify successful programming
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (1<<23)|(verification_register<<16)) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (2 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()
        status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/STATUS')
        if (status & 0x0001):
            # data readout has no meaning here as we were not asking any, just keep it for consistency
            rd_data = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/READDATA')
            if rd_data != reg[verification_register]&0x00FFFF:
                print(" Verification of Register %d unsuccessful: %x" %(verification_register, rd_data))
        else:
            print(" SPI access failed: 0x%04X\n" % (status))

        # read out PLL Lock register
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (1<<23)|(110<<16)) # enable channel SPIM0_OE3
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (2 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
        Grimsel.daq.sync()
        status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/STATUS')
        if (status & 0x0001):
            # data readout has no meaning here as we were not asking any, just keep it for consistency
            rd_data = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/READDATA')
            if ((rd_data&0x0600)>>9) != 2:              # check whether PLL could lock or not
                print(" PLL couldn't lock!")
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan) # PS_SPI0_SS0 -> SPIM0_OEx
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', 0xFF) # enable channel SPIM0_OE3
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (1 << 0)) # nothing to read, 2 bytes to write
            Grimsel.daq.sync()
        else:
            print(" SPI access failed: 0x%04X\n" % (status))

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan) # PS_SPI0_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', 0xFF) # enable channel SPIM0_OE3
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (1 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()
    return PLL_Power

def UINT2FLOAT(integer_value):
    return struct.unpack('!f', struct.pack('!I', integer_value))[0]

def FLOAT2UINT(float_value):
    return struct.unpack('!I', struct.pack('!f', float_value))[0]

def POWER2UINT_IQ(rel_power):
    assert 0 <= rel_power <=1, " rel_power must be between 0 and 1"
    i_value = np.sqrt(rel_power)
    i_uint = int(i_value*0x7FFF)
    return i_uint + (i_uint << 16)

def setDAC(channel=0, freq=2.0, power_dbfs=-80, op='single', dist = 10, noise_filter = 'on'):
    #random_filter_enable    = True # whether or not to filter the noise for reducing the BW.

    # prevent higher two-tone offsets than 100 MHz
    if dist >= 100:
        dist = 100

    # limit DAC output power to safe level in case a dual tone output is requested
    if (power_dbfs >= -9) & (op == 'dual'):
        power_dbfs = -9

    # There are 4 so-called DACSOURCES, which can be switched between sine wave, AWG, or random signal
    # Each DACSOURCE is connected to a pair of neighbouring DACs.
    # Seen from the front panel, DACSOURCE 0 serves the two leftmost DACs and DACSOURCE 3 serves the two rightmost DACs
    dacsource_idx = int(channel/2)

    # The DACSOURCE path index determines which of the two output channels served by each DACSOURCE should be configured:
    dacsource_path_idx = channel % 2

    if (power_dbfs <= -80):
        dacsource_en = 0
        print(" Disabling DAC #%d"%(channel))
    else:
        dacsource_en = 1
        print(" Setting DAC #%d to %f GHz with %d dBFS" %(channel, freq, power_dbfs))

    Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DACSOURCES/{dacsource_idx}/PATHS/{dacsource_path_idx}/ON', dacsource_en)

    # In[]: Set the output power
    #
    # Set I/Q values according to a given power
    # Example: rel_power = 1.0 corresponds to a sine wave with full-range amplitude
    # Example: rel_power = 0.25 corresponds to a sine wave with half-range amplitude
    rel_power = 10**(power_dbfs/10)
    rel_power2 = 10**((power_dbfs+9)/20)

    iq_value = POWER2UINT_IQ(rel_power)

    # Two-tone support
    # IMPORTANT:  set the power of the first tone maximally to -9 dBFS !
    #             Otherwise the signal will suffer from clipping!
    # Set sines_gain to 0, to disable the second tone.
    if op == 'dual':
        sines_gain  = int(rel_power2*32767)  # gain of the second tone (0x10000 corresponds to a gain of 1)
        osc_freq    = dist*1e6   # oscillator frequency for two-tone support
        freq        = freq-(dist/(2*1e3))
        # limit output power to -9dBFS to prevent high spur levels
    else:
        sines_gain = 0x00000
        osc_freq    = 0   # oscillator frequency for two-tone support

    # Loop to set all parallel paths to a constant I/Q value
    for i in range(6):
        addr = i + DEBUG_REG_REG_IQ_VALUES_OFFSET32
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/{addr}/VALUE', iq_value)

    # In[124]: Set mixer frequencies
    mixer_index         = channel
    mixer_frequency_hz = freq*1e9
    addr = DEBUG_REG_REG_MIXER_FREQUENCIES_OFFSET32 + mixer_index
    Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/{addr}/VALUE', FLOAT2UINT(mixer_frequency_hz))

    #if op == 'dual':
    #%% Enable a second tone (two-tone support for intermodulation measurements)
    #print ("SINES gain: {:05x}".format(sines_gain))
    addr = DEBUG_REG_REG_OUTPUT_SINES_GAIN
    Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/{addr}/VALUE', sines_gain)

    #%% Oscillator frequency for two-tone support
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/OSCS/0/FREQ', osc_freq)

    #%% Random source enable if requested
    if op == 'noise':
        addr = DEBUG_REG_REG_DAC_MIXER_TEST_ID
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/{addr}/VALUE', 2) # bypass mixer

        print(f'enabling the noise output {dacsource_idx}')
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DACSOURCES/{dacsource_idx}/SOURCESELECT', 2)
    else:
        addr = DEBUG_REG_REG_DAC_MIXER_TEST_ID

        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/{addr}/VALUE', 2) # enable fine mixer
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DACSOURCES/{dacsource_idx}/SOURCESELECT', 0)

    return

def setLatchEnable(channel, state):
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', channel) # PS_SPI1_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', ~(state << 4)) # enable channel SPIM1_OE4
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (1 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()
    return

def set_ADF4002(r_cnt = 10, n_cnt = 1, cp_gain = 0, cp_gain1 = 0b111, cp_gain2 = 0b111, pfd_pol = 1, pd = 0, fastlock = 0, fastlock_mode = 0, abp_width = 'slow'):
    SPI_chan = 3

    if abp_width == 'fast':
        abpw = 0
    else:
        abpw = 2

    ADF4002_r_cnt_latch = (0b000<<21)|(1<<20)|(0b00<<18)|(abpw<<16)|(int(r_cnt)<<2)|(0b00)
    ADF4002_n_cnt_latch = (0b00<<22)|(cp_gain<<21)|(int(n_cnt)<<8)|(0b000000<<2)|(0b01)
    ADF4002_funct_latch = (0b00<<22)|(0<<21)|(cp_gain2<<18)|(cp_gain1<<15)|(0b1111<<11)|(fastlock_mode<<10)|(fastlock<<9)|(0<<8)|(pfd_pol<<7)|(0b001<<4)|(pd<<3)|(0<<2)|(0b10)
    ADF4002_init_latch = (ADF4002_funct_latch)|(0b11)

    setLatchEnable(SPI_chan, 1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', ADF4002_init_latch) # enable channel SPIM0_OE3
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()
    #print("Writing 0x%4X to SPI Slave"%(ADF4002_init_latch))
    setLatchEnable(SPI_chan, 0)

    setLatchEnable(SPI_chan, 1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', ADF4002_funct_latch) # enable channel SPIM0_OE3
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()
    #print("Writing 0x%4X to SPI Slave"%(ADF4002_funct_latch))
    setLatchEnable(SPI_chan, 0)

    setLatchEnable(SPI_chan, 1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', ADF4002_r_cnt_latch) # enable channel SPIM0_OE3
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()
    #print("Writing 0x%4X to SPI Slave"%(ADF4002_r_cnt_latch))
    setLatchEnable(SPI_chan, 0)

    setLatchEnable(SPI_chan, 1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI0_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', ADF4002_n_cnt_latch) # enable channel SPIM0_OE3
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (3 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()
    #print("Writing 0x%4X to SPI Slave"%(ADF4002_n_cnt_latch))
    setLatchEnable(SPI_chan, 0)

def setIntRef():
    print(" Setting internal reference")
    setGPO(4,0)
    setFP_LED('extref','off')
    setFP_LED('zsync','off')
    set_ADF4002(pd = 1)
    return

def setExtRef(input='ext', ref_in=10e6, pfd_freq=10e6, timeout=1.5, cpg=5):
    pll_locked = 0

    if input=='ext':
        led = 'extref'
    else:
        led = 'zsync'

    setFP_LED(led, 'yellow')

    #check if PFD freq is above 50MHz to change Antibacklash Pulse Width according to DS
    if pfd_freq>50e6:
        abpw = 'fast'
    else:
        abpw = 'slow'

    print(" Configuring PLL for external frequency of %0.1fMHz with %dkHz PFD" %(ref_in/1e6, pfd_freq/1e3))
    # do the programming in the right order as described in DS, p.16
    set_ADF4002(r_cnt = ref_in/pfd_freq, n_cnt = 100e6/pfd_freq, cp_gain = 0, cp_gain1 = cpg, cp_gain2 = 0b011, pfd_pol = 1, pd = 0, fastlock = 0, fastlock_mode = 0, abp_width = abpw)

    if input == 'ext':              # ExtRef
        setGPO(3,1)
        setGPO(4,1)
    else:                           # ZSync
        setGPO(3,0)
        setGPO(4,1)

    start_time = time.time()        # memorize start time (=now)

    while time.time() <= start_time+timeout:
        if getGPI(2):               # checking whether the PLL could lock or not
            pll_locked = 1
            break

    if pll_locked:
        print(" Successfully locked to external reference")
        #time.sleep(1)
        setFP_LED(led, 'green')

    else:
        print(" Locking to external source failed, reverting to Internal source")
        setFP_LED(led, 'red')
        setIntRef()
    return

def LTC2991_code_to_single_ended_voltage(adc_code):
    if adc_code >> 14:
        adc_code = (adc_code ^ 0x7FFF) + 1                 #! 1) Converts two's complement to binary
        sign = -1
    else:
        sign=1
    adc_code = (adc_code & 0x3FFF)
    return float(adc_code) * LTC2991_SINGLE_ENDED_lsb * sign   #! 2) Convert code to voltage from lsb

def scaleVoltages(LTC2991_input,a_type):
    V_SCALING_P10V = float(2400+787)/787
    V_SCALING_P5V = float(1200+1200)/1200
    V_SCALING_P3V75 = float(1200+2400)/2400
    V_SCALING_P3V3 = float(1200+3600)/3600
    V_SCALING_P2V5 = 1
    V_SCALING_P1V8 = 1
    CUR_SCALING = 5/3
    LTC2991_input[0] = LTC2991_input[0]
    LTC2991_input[1] = LTC2991_input[1] * V_SCALING_P3V3
    LTC2991_input[2] = LTC2991_input[2] * V_SCALING_P3V75
    LTC2991_input[3] = LTC2991_input[3] * V_SCALING_P3V3
    LTC2991_input[4] = LTC2991_input[4] * V_SCALING_P10V
    LTC2991_input[5] = LTC2991_input[5] * V_SCALING_P2V5
    LTC2991_input[6] = LTC2991_input[6] * V_SCALING_P10V
    LTC2991_input[7] = LTC2991_input[7] * V_SCALING_P1V8
    LTC2991_input[8] = LTC2991_input[8] * V_SCALING_P5V
    LTC2991_input[9] = LTC2991_input[9] - float(5 - LTC2991_input[9]) / 1100 * 3300
    LTC2991_input[10] = LTC2991_input[10] - float(5 - LTC2991_input[10]) / 1100 * 2200
    LTC2991_input[11] = LTC2991_input[11] - float(5 - LTC2991_input[11]) / 1200 * 1800
    if a_type =='up':
        LTC2991_input[12] = LTC2991_input[12] * CUR_SCALING
        LTC2991_input[13] = LTC2991_input[13] * CUR_SCALING
    else:
        LTC2991_input[12] = LTC2991_input[12] * V_SCALING_P5V
        LTC2991_input[13] = LTC2991_input[13] * V_SCALING_P5V
    LTC2991_input[14] = LTC2991_input[14] * CUR_SCALING
    LTC2991_input[15] = LTC2991_input[15] * CUR_SCALING

    return LTC2991_input


def readSYSMON(channel,a_type,a_print=True):
    LTC2991_voltage=[np.NaN]*16
    if a_type== 'up':
        LTC2991_Unit = ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "A", "A", "A", "A"]
    elif a_type=='down':
        LTC2991_Unit = ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "A", "A"]
    else:
        LTC2991_Unit = ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"]
    frontend = getPCBRevision(i2c_mux=channel+1, i2c_addr=MCP23017_addr_frontend)[0]
    addr = [LTC2991_addr_1, LTC2991_addr_2_alt] if (frontend & 0xF00F) == 0x0002 else [LTC2991_addr_1, LTC2991_addr_2]

    Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1) # channel 10, but we count from 1 as 0 means no channel
    for udx,address in enumerate(addr):
        #configure
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', address)
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (LTC2991_reg_vcc_tint<<8)|(LTC2991_vcc_tint<<0))
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (2 << 0)) # 2 bytes to read, 1 byte to write
        Grimsel.daq.sync()

        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', address)
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (LTC2991_reg_status_hi<<8)|(LTC2991_status_hi<<0))
        Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (2 << 0)) # 2 bytes to read, 1 byte to write
        Grimsel.daq.sync()

        #print("LTC2991 0x%02X" %(address))
        #print("--------------")
        for x in range(8):
            Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', address)
            Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (LTC2991_reg_v1)+2*x)
            Grimsel.daq.setInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
            Grimsel.daq.sync()
            status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
            if status & 0x0001:
                bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA') & 0x7FFF
                LTC2991_voltage[x+udx*8]=LTC2991_code_to_single_ended_voltage(bin_temp)
            else:
                print("LTC2991 readout failed!")

    LTC2991_Val=scaleVoltages(LTC2991_voltage,a_type)
    if a_print:
        for idx,val in enumerate(LTC2991_Val):
            print("V%1d.%1d:  %02.3f %s" %(idx/8+1, math.fmod(idx,8)+1,val, LTC2991_Unit[idx]))

    return LTC2991_Val,LTC2991_Unit



def setLMH6401(channel=0, gain=0, pd=0):
    SPI_chan = 0

    #channel: # Modulo 2 = 0: BB-LMH, Modulo 2 = 1: AUX LMH
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan) # PS_SPI1_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', 1 << channel) # enable channel SPIM1_OE4
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (2 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI1_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (1<<15)|(LMH6401_addr_revID<<8)) # enable channel SPIM1_OE4
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (1 << 8) | (2 << 0)) # 1 byte to read, 1 byte to write
    Grimsel.daq.sync()
    status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/SPI/STATUS')
    if (status & 0x0001):
        # data readout has no meaning here as we were not asking any, just keep it for consistency
        rd_data = Grimsel.daq.getDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/READDATA')
        if rd_data==0x03:
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan+1) # PS_SPI1_SS0 -> SPIM0_OEx
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', (0<<15)|(LMH6401_addr_gain<<8)|(pd<<6)|(26-gain)) # enable channel SPIM1_OE4
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (2 << 0)) # 1 byte to read, 1 byte to write
            Grimsel.daq.sync()
            #print("Gain set to %ddB" %(gain))
        #else:
        #    print("Read data: 0x%02X\n" % (rd_data))
    else:
        print("SPI access failed: 0x%04X\n" % (status))
    # deselecting the SPI slave select at last
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/BUS', SPI_chan) # PS_SPI0_SS0 -> SPIM0_OEx
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/WRITEDATA', 0 << channel) # enable channel SPIM0_OE3
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/SPI/TRANSFER', (0 << 8) | (2 << 0)) # nothing to read, 2 bytes to write
    Grimsel.daq.sync()
    return

def setDownconverter(channel=1, path='rf', rf_att=9.5, rf_pre='off', rf_postamp=15, cal='off', bb_gain=-6, bb_att='off', bb_filt='off', trg_z='high', trg_loop='off'):
    if rf_pre in ['off', 'on']:
        rf_pre= (rf_pre == 'on')
    if bb_att in ['off', 'on']:
        bb_att= (bb_att == 'on')
    if bb_filt in ['off', 'on']:
        bb_filt= (bb_filt == 'on')
    CAL_DAC_dc = 0x0        # 0 = Off 1 = On
    CAL_ADC_dc = 0x0        # 0 = Off 1 = On
    CAL_REF_dc = 0x0        # 0 = Off 1 = On
    if path=='rf':
        RF_BB_IN_dc = 0x0       # 0 = RF 1 = BB
        IN_SW =  0b00001000      # 0 = Off, 0b00001000 = RF Path,  0b00100000 = BB Path
        BB_LMH_PD_dc = 0x1      # 0 = LMH On 1 = LMH Off
    elif path=='bb':
        RF_BB_IN_dc = 0x1       # 0 = RF 1 = BB
        IN_SW = 0b00100000      # 0 = Off, 0b00001000 = RF Path,  0b00100000 = BB Path
        BB_LMH_PD_dc = 0x0      # 0 = LMH On 1 = LMH Off
    else:
        RF_BB_IN_dc = 0x0       # 0 = RF 1 = BB
        IN_SW =  0b00000000     # 0 = Off, 0b00001000 = RF Path,  0b00100000 = BB Path
        BB_LMH_PD_dc = 0x1      # 0 = LMH On 1 = LMH Off

    if cal=='on':
        BIST_SW = 0b01000000    # 0 = Off, 0b00000001 = RF Input, 0b01000000 = DAC_BIST
    elif cal=='term':
        BIST_SW = 0b00000000    # 0 = Off, 0b00000001 = RF Input, 0b01000000 = DAC_BIST
        IN_SW = 0b00000000  # 0 = Off, 0b00001000 = RF Path,  0b00100000 = BB Path
    else:
        BIST_SW = 0b00000001    # 0 = Off, 0b00000001 = RF Input, 0b01000000 = DAC_BIST

    if rf_pre:
        RF_PREAMP = 0x1         # 0 = Off, 1 = On
    else:
        RF_PREAMP = 0x0         # 0 = Off, 1 = On

    if trg_loop=='on':
        TRG_LOOPBACK = 0x1         # 0 = Off, 1 = On
    else:
        TRG_LOOPBACK = 0x0         # 0 = Off, 1 = On

    if bb_filt:
        BB_BW = 0x1             # 0 = Full, 1 = 2 GHz
    else:
        BB_BW = 0x0             # 0 = Full, 1 = 2 GHz

    if bb_att:
        BB_Att = 0x1            # 0 = Off, 1 = On
    else:
        BB_Att = 0x0            # 0 = Off, 1 = On

    if trg_z=='high':
        TRG_IN_ATT_dc = 0x0     # 0 = 1k 1 = 50R
    else:
        TRG_IN_ATT_dc = 0x1     # 0 = 1k 1 = 50R

    ATT_VAL_dc = reverseBits(int(rf_att*2), 6)

    TCA6424_downconv_out0 = (TRG_LOOPBACK << 7) | (RF_PREAMP << 6) | (ATT_VAL_dc << 0)
    TCA6424_downconv_out1 = (BB_Att<<7)|(BB_LMH_PD_dc<<6)|(BB_BW<<5)|(RF_BB_IN_dc<<4)|(CAL_REF_dc<<3)|(CAL_ADC_dc<<2)|(CAL_DAC_dc<<1)|(TRG_IN_ATT_dc<<0)
    TCA6424_downconv_out2 = (BIST_SW)|(IN_SW)

    setLMH6401(channel*2,bb_gain,0)

    print(" Configuring Downconverter in Slot %01d..." %(channel))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_frontend)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', 0x00FF0F)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_frontend)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', 0x120000)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()

    #print("- TCA6424A Output array content: 0x%02X 0x%02X 0x%02x" % (TCA6424_downconv_out0,TCA6424_downconv_out1,TCA6424_downconv_out2))

    # setting TCA6424 direction register
    # setting TCA6424A Output direction register to all outputs
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', TCA6424_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (TCA6424_reg_iodir<<24) | (TCA6424_iodir<<0))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (4 << 0))
    Grimsel.daq.sync()
#
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', TCA6424_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (TCA6424_reg_out <<24) | (TCA6424_downconv_out0<<16) | (TCA6424_downconv_out1<<8) | (TCA6424_downconv_out2<<0) )
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (4 << 0))
    Grimsel.daq.sync()
    # divide up requested attenuation into coarse and fine
    if rf_postamp < 5:
        p5dB = 1
        p10dB = 1
    elif 5 <= rf_postamp < 10:
        p5dB = 0
        p10dB = 1
    elif 10 <= rf_postamp < 15:
        p5dB = 1
        p10dB = 0
    elif rf_postamp >=15:
        p5dB = 0
        p10dB = 0
    else:
        p5dB = 1
        p10dB = 1

    MCP23017_frontend_value = (0x12 << 16) | (0x00 << 8) | (p5dB<< 5) | (p10dB << 6) | (0x0)

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_frontend)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', MCP23017_frontend_value)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()


    print(" ...done")
    return

def setUpconverter(channel=0, freq=2.0, rf_disable=1, path='rf', rf_att=9.5, bb_gain=-6, bb_att='off', aux='off', aux_gain=-6, aux_att='off', rf_tap='off', trg_z='high'):

    if freq <= 1.5:
        rf_filter = 0x00
    elif (freq > 1.5) & (freq <= 3):
        rf_filter = 0x01
    elif (freq > 3) & (freq < 5.6):
        rf_filter = 0x02
    else:
        rf_filter = 0x03

    # confine attenuation value to valid levels
    if rf_att > 66.5:
        rf_att = 66.5
    elif rf_att < 0:
        rf_att = 0

    # divide up requested attenuation into coarse and fine
    if 5 <= rf_att < 10:
        rf_attcrs=5
        rf_att_res = rf_att - 5
    elif 10 <= rf_att < 15:
        rf_attcrs=10
        rf_att_res = rf_att - 10
    elif rf_att >=15:
        rf_attcrs=15
        rf_att_res = rf_att - 15
    else:
        rf_attcrs=0
        rf_att_res = rf_att

    # set fine attenuation but activate 20dB bulk attenuator if necessary
    if rf_att_res <=31.5:
        rf_attfine = int(rf_att_res*2)/2
        rf_att20db = False
    else:
        rf_attfine = int((rf_att_res-20)*2)/2
        rf_att20db = True

    setUpconverter_adv(channel=channel, rf_disable=rf_disable, path=path, rf_attcrs=rf_attcrs, rf_att20db=rf_att20db, rf_attfine=rf_attfine, rf_filter=rf_filter, bb_gain=bb_gain, bb_att=bb_att, aux=aux, aux_gain=aux_gain, aux_att=aux_att,
                       rf_tap=rf_tap, trg_z=trg_z)
    return

def setUpconverter_adv(channel=0, rf_disable=1, bb_disable=1, path='rf', rf_attcrs=15, rf_att20db=True , rf_attfine=31, rf_filter=0, bb_gain=-6, bb_att='off', aux='off', aux_gain=-6, aux_att='off', rf_tap='off', trg_z='high', dac_inner_loop='off', adc_inner_loop='off', auxin_outer_loop=None, rf_var_amp=None, rf_fixed_att=None, rf_var_att_override=None):

    if dac_inner_loop=='on':
        CAL_DAC_uc = 0x1        # 0 = Off 1 = On
    else:
        CAL_DAC_uc = 0x0        # 0 = Off 1 = On

    if adc_inner_loop=='on':
        CAL_ADC_uc = 0x1        # 0 = Off 1 = On
    else:
        CAL_ADC_uc = 0x0        # 0 = Off 1 = On

    CAL_REF_uc = 0x0        # 0 = Off 1 = On

    led_clip = 0
    p10dB = 1
    p5dB = 1
    if bb_att in ['off', 'on']:
        bb_att= (bb_att == 'on')
    if aux in ['off', 'on']:
        aux= (aux == 'on')
    if aux_att in ['off', 'on']:
        aux_att= (aux_att == 'on')
    if rf_tap in ['off', 'on']:
        rf_tap = (rf_tap == 'on')

    # configure GPIO output variables according to configuration
    if rf_disable==1:
        led_rf = 0
    else:
        led_rf = 1

    if path=='rf':
        RF_BB_IN_uc = 0x0       # 0 = RF 1 = BB
        OUT_SW_2_1 = 0x0        # 0 = Off 1 = BB
        OUT_SW_2_2 = 0x1        # 0 = Off 1 = RF
        BB_LMH_PD_uc = 0x1      # 0 = LMH On 1 = LMH Off
        RF_PA_EN = rf_disable
    elif path=='bb':
        RF_BB_IN_uc = 0x1       # 0 = RF 1 = BB
        OUT_SW_2_1 = 0x1        # 0 = Off 1 = BB
        OUT_SW_2_2 = 0x0        # 0 = Off 1 = RF
        BB_LMH_PD_uc = 0x0      # 0 = LMH On 1 = LMH Off
        RF_PA_EN = 0x1
    elif path=='term':
        RF_BB_IN_uc = 0x1       # 0 = RF 1 = BB
        OUT_SW_2_1 = 0x1        # 0 = Off 1 = BB
        OUT_SW_2_2 = 0x0        # 0 = Off 1 = RF
        BB_LMH_PD_uc = rf_disable      # 0 = LMH On 1 = LMH Off
        RF_PA_EN = 0x1
    else:
        RF_BB_IN_uc = 0x0       # 0 = RF 1 = BB
        OUT_SW_2_1 = 0x0        # 0 = Off 1 = BB
        OUT_SW_2_2 = 0x1        # 0 = Off 1 = RF
        BB_LMH_PD_uc = 0x1      # 0 = LMH On 1 = LMH Off
        RF_PA_EN = 0x1

    if bb_att:
        BB_POST_ATT = 0x1       # 0 = Off 1 = On
    else:
        BB_POST_ATT = 0x0

    if aux_att:
        AUX_IN_ATT = 0x1        # 0 = Off 1 = On
    else:
        AUX_IN_ATT = 0x0        # 0 = Off 1 = On

    if aux:
        AUX_LMH_PD_uc = 0x0     # 0 = LMH On 1 = LMH Off
    else:
        AUX_LMH_PD_uc = 0x1     # 0 = LMH On 1 = LMH Off

    if rf_tap:
        OUT_SW_1_1 = 0x0        # 0 = Off 1 = RF OUT
        OUT_SW_1_2 = 0x1        # 0 = Off 1 = RF Tap
        AUX_IN_SEL = 0x1        # 0 = AUX 1 = RF Tap
    else:
        OUT_SW_1_1 = 0x1        # 0 = Off 1 = RF OUT
        OUT_SW_1_2 = 0x0        # 0 = Off 1 = RF Tap
        AUX_IN_SEL = 0x0        # 0 = AUX 1 = RF Tap

    # drive auxin outer loopback relay independently from the rf_tap parameter
    if auxin_outer_loop=='on':
        AUX_IN_SEL = 0x1        # 0 = AUX 1 = RF Tap
    elif auxin_outer_loop=='off':
        AUX_IN_SEL = 0x0        # 0 = AUX 1 = RF Tap

    if trg_z=='high':
        TRG_IN_ATT_uc = 0x0     # 0 = 1k 1 = 50R
    else:
        TRG_IN_ATT_uc = 0x1     # 0 = 1k 1 = 50R

    FILTER_BANK = rf_filter

    # confine attenuation value to valid levels
    if rf_attcrs+rf_att20db+rf_attfine > 66.5:
        rf_attcrs = 15
        rf_attfine = 31.5
        rf_att20db = True
    elif rf_attcrs+rf_att20db+rf_attfine < 0:
        rf_attcrs = 0
        rf_attfine = 0
        rf_att20db = False

    # divide up requested attenuation into coarse and fine
    if 5 <= rf_attcrs < 10:
        p5dB = 0
        p10dB = 1
    elif 10 <= rf_attcrs < 15:
        p5dB = 1
        p10dB = 0
    elif rf_attcrs >=15:
        p5dB = 0
        p10dB = 0
    else:
        p5dB = 1
        p10dB = 1

    # set fine attenuation but activate 20dB bulk attenuator if necessary
    ATT_VAL_uc = reverseBits(int(rf_attfine*2), 6)
    RF_ATT = int(not(rf_att20db))            # 0 = On 1 = Off

    # override variable gain stage if set
    if rf_var_amp==0:
        p5dB = 0
        p10dB = 0
    elif rf_var_amp==5:
        p5dB = 1
        p10dB = 0
    elif rf_var_amp==10:
        p5dB = 0
        p10dB = 1
    elif rf_var_amp==15:
        p5dB = 1
        p10dB = 1

    # override coarse attenuator if set
    if rf_fixed_att=='on':
        RF_ATT = 0x0            # 0 = On 1 = Off
    elif rf_fixed_att=='off':
        RF_ATT = 0x1            # 0 = On 1 = Off

    # override variable attenuator if set
    if rf_var_att_override is not None:
        if rf_var_att_override>=0 and rf_var_att_override<=31:
            ATT_VAL_uc = reverseBits(int(rf_var_att_override*2), 6)

    TCA6424_upconv_out0 = (OUT_SW_1_2 << 7) | (OUT_SW_1_1 << 6) | (ATT_VAL_uc << 0)
    TCA6424_upconv_out1 = (CAL_ADC_uc<<7) | (AUX_IN_ATT<<6) | (RF_BB_IN_uc<<5) | (RF_PA_EN<<4) | (FILTER_BANK << 2) | (OUT_SW_2_2 << 1) | (OUT_SW_2_1 << 0)
    TCA6424_upconv_out2 = (CAL_DAC_uc << 7) | (CAL_REF_uc<<6) | (AUX_IN_SEL<<5) | (AUX_LMH_PD_uc<<4) | (TRG_IN_ATT_uc<<3) | (RF_ATT<<2) | (BB_LMH_PD_uc<<1) | (BB_POST_ATT<<0)

    print(" Configuring Upconverter in Slot %01d..." %(channel))
    setLMH6401(channel*2,bb_gain,0)
    setLMH6401(channel*2+1,aux_gain,0)

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_frontend)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', 0x00FF0F)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()

    MCP23017_frontend_value = (0x12 << 16) | (0x00 << 8) | (p10dB << 7) | (p5dB<< 6) | (led_clip << 5) | (led_rf << 4) | (0x0)

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_frontend)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', MCP23017_frontend_value)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()
    print("...done")

    #print("TCA6424A Output array content: 0x%02X 0x%02X 0x%02x" % (TCA6424_upconv_out0,TCA6424_upconv_out1,TCA6424_upconv_out2))

    # setting TCA6424 direction register
    # setting TCA6424A Output direction register to all outputs
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', TCA6424_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (TCA6424_reg_iodir<<24) | (TCA6424_iodir<<0))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (4 << 0))
    Grimsel.daq.sync()
#
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', channel+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', TCA6424_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (TCA6424_reg_out <<24) | (TCA6424_upconv_out0<<16) | (TCA6424_upconv_out1<<8) | (TCA6424_upconv_out2<<0) )
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (4 << 0))
    Grimsel.daq.sync()
    return

def initFanController():
    MAX31790_fans = [MAX31790_fan_CASE, MAX31790_fan_CASE, MAX31790_fan_CASE, MAX31790_fan_FPGA, MAX31790_fan_CASE, MAX31790_fan_PSU]

    for i in range(6):
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', MAX31790_mux+1)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MAX31790_addr)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', ((MAX31790_reg_config_fan1+i)<<8)|(MAX31790_fans[i]<<0))
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (2 << 0)) # 2 bytes to read, 1 byte to write
        Grimsel.daq.sync()
    return

def getFanSpeeds():
    # returns actual fan speeds of case fans, psu fan and fpga fan
    temp = 0
    fan_index = [1, 3, 5]
    fan_speeds = [0, 0, 0]

    for i in fan_index:
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', MAX31790_mux+1)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MAX31790_addr)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (MAX31790_reg_tach1+2*i))
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
        Grimsel.daq.sync()
        status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
        if (status & 0x0001):
            fan_speeds[temp] = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA') >> 5
            #print("Readback from MAX31790: 0x%02X" %(bin_temp))
        else:
            print("Readback failed!")
        temp += 1

    fan_speeds[0] = int(8192/fan_speeds[0]*60*2)
    fan_speeds[1] = int(8192/fan_speeds[1]*60*2)
    fan_speeds[2] = int(8192/fan_speeds[2]*60*2)

    return fan_speeds[0], fan_speeds[1], fan_speeds[2]

def getFans():
    # returns actual PWM speed of case fans, psu fan and fpga fan
    temp = 0
    fan_index = [1, 3, 5]
    pwm_speeds = [0, 0, 0]
    for i in fan_index:
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', MAX31790_mux+1)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MAX31790_addr)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (MAX31790_reg_PWM1_current+2*i))
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
        Grimsel.daq.sync()
        status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
        if (status & 0x0001):
            pwm_speeds[temp] = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA') >> 7
            #print("Readback from MAX31790: 0x%02X" %(bin_temp))
        else:
            print("Readback failed!")
        temp += 1
    return pwm_speeds[0], pwm_speeds[1], pwm_speeds[2]

def setFans(case_speed=255, psu_speed=192, fpga_speed=255):

    MAX31790_PWM_PSU = psu_speed       # 0-511 = 0-100% Duty Cycle
    MAX31790_PWM_CASE = case_speed     # 0-511 = 0-100% Duty Cycle
    MAX31790_PWM_FPGA = fpga_speed     # 0-511 = 0-100% Duty Cycle
    MAX31790_fans_PWM = [MAX31790_PWM_CASE, MAX31790_PWM_CASE, MAX31790_PWM_CASE, MAX31790_PWM_FPGA, MAX31790_PWM_CASE, MAX31790_PWM_PSU]

    for i in range(6):
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', MAX31790_mux+1)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MAX31790_addr)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', ((MAX31790_reg_PWM1_target+2*i)<<16)|(MAX31790_fans_PWM[i]<<7))
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
        Grimsel.daq.sync()
    return

def setFreezeAll():
    freeze=Grimsel.freeze
    freeze_setting = 0
    # 0 = not frozen
    # 1 = freeze
    freeze[0] = 1
    freeze[1] = 1
    freeze[2] = 1
    freeze[3] = 1
    freeze[4] = 1
    freeze[5] = 1
    freeze[6] = 1
    freeze[7] = 1
    #for i in range(8):
    freeze_setting = (int(freeze[0]) << 7)|(int(freeze[1]) << 6)|(int(freeze[2]) << 5)|(int(freeze[3]) << 4)|(int(freeze[4]) << 3)|(int(freeze[5]) << 2)|(int(freeze[6]) << 1)|(int(freeze[7]))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/16/VALUE', freeze_setting)
    Grimsel.daq.sync()
    Grimsel.freeze=freeze

    return

def setFreeze(chan, val):
    freeze=Grimsel.freeze
    freeze_setting = 0
    # 0 = not frozen
    # 1 = freeze
    freeze[chan] = val
    freeze_setting = (int(freeze[0]) << 7)|(int(freeze[1]) << 6)|(int(freeze[2]) << 5)|(int(freeze[3]) << 4)|(int(freeze[4]) << 3)|(int(freeze[5]) << 2)|(int(freeze[6]) << 1)|(int(freeze[7]))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/16/VALUE', freeze_setting)
    Grimsel.daq.sync()
    Grimsel.freeze=freeze

    return

def setNyquistAll():
    zones = Grimsel.zones
    # 0 = even zone (for IF = 3 GHz)
    # 1 = odd zone  (for baseband)
    zones[0] = 1
    zones[1] = 0
    zones[2] = 1
    zones[3] = 0
    zones[4] = 1
    zones[5] = 0
    zones[6] = 1
    zones[7] = 0
    nyquist_zone_setting = (int(zones[0]) << 7)|(int(zones[1]) << 6)|(int(zones[2]) << 5)|(int(zones[3]) << 4)|(int(zones[4]) << 3)|(int(zones[5]) << 2)|(int(zones[6]) << 1)|(int(zones[7]))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/17/VALUE', nyquist_zone_setting)
    Grimsel.zones=zones
    Grimsel.nyquist_zone_setting=nyquist_zone_setting
    return

def setNyquist(chan, val):
    zones=Grimsel.zones
    zones[chan] = val
    Grimsel.nyquist_zone_setting = (int(zones[0]) << 7)|(int(zones[1]) << 6)|(int(zones[2]) << 5)|(int(zones[3]) << 4)|(int(zones[4]) << 3)|(int(zones[5]) << 2)|(int(zones[6]) << 1)|(int(zones[7]))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/17/VALUE', Grimsel.nyquist_zone_setting)
    Grimsel.zones=zones
    return

def initFP():
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', 0)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_FP)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', 0x000000)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0))
    Grimsel.daq.sync()
    return

def getFP():
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', 0)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_FP)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', 0x12)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0))
    Grimsel.daq.sync()
    status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
    if (status & 0x0001):
        bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
        #print(" Readback: %x" % (bin_temp))
        output_array1 = ((bin_temp&0x00FF) << 8)
        output_array2 = ((bin_temp&0xFF00) >> 8)
        output_array = (output_array1)|(output_array2)

        colors[0] = (output_array & 0x7000) >> 12
        colors[1] = (output_array & 0x0E00) >> 9
        colors[2] = (output_array & 0x01C0) >> 6
        colors[3] = (output_array & 0x0038) >> 3
    else:
        print(" Readback failed")

    return

def setFP_LED(name='zsync', color='off'):
    getFP()

    if name=='zsync':
        x=1
    elif name=='extref':
        x=2
    elif name=='busy':
        x=3
    else:
        x=0

    if color=='green':
        colors[x] = 0x02
    elif color=='red':
        colors[x] = 0x04
    elif color=='blue':
        colors[x] = 0x01
    elif color=='cyan':
        colors[x] = 0x03
    elif color=='yellow':
        colors[x] = 0x06
    elif color=='purple':
        colors[x] = 0x05
    elif color=='white':
        colors[x] = 0x07
    else:
        colors[x] = 0x00

    output_array = (colors[0]<<12)|(colors[1]<<9)|(colors[2]<<6)|(colors[3]<<3)|(0x00<<0)
    output_array1 = ((output_array&0x00FF) << 8)
    output_array2 = ((output_array&0xFF00) >> 8)
    output_array = (output_array1)|(output_array2)

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', 0)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_FP)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (0x12<<16)|output_array)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()
    return

def setFP_LED_All(status='green', zsync='off', extref='off', busy='off'):

    colors[0] = status
    colors[1] = zsync
    colors[2] = extref
    colors[3] = busy
    x=0

    for color in colors:
        if color=='green':
            colors[x] = 0x02
        elif color=='red':
            colors[x] = 0x04
        elif color=='blue':
            colors[x] = 0x01
        elif color=='cyan':
            colors[x] = 0x03
        elif color=='yellow':
            colors[x] = 0x06
        elif color=='purple':
            colors[x] = 0x05
        elif color=='white':
            colors[x] = 0x07
        else:
            colors[x] = 0x00
        x+=1

    output_array = (colors[0]<<12)|(colors[1]<<9) | (colors[2]<<6)|(colors[3]<<3)|(0x00<<0)
    output_array1 = ((output_array&0x00FF) << 8)
    output_array2 = ((output_array&0xFF00) >> 8)
    output_array = (output_array1)|(output_array2)

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', 0)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', MCP23017_addr_FP)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (0x12<<16)|output_array)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()
    return

def readTemp(mux,read_addr,num_of_sensors):
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', mux+1) # channel 10, but we count from 1 as 0 means no channel
    temp = []
    for addr_temp_sens in range(read_addr, read_addr+num_of_sensors):
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', addr_temp_sens)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', 0x05)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
        Grimsel.daq.sync()
        status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
        if (status & 0x0001):
            bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
            ftemp = float(bin_temp & 0x0fff) * (1.0/(1 << 4))
            if (bin_temp & 0x1000):
                ftemp *= -1.0
            print(" Temperature sensor 0x%02X ambient: %0.3f" % (addr_temp_sens, ftemp))
            temp.append(ftemp)
        else:
            print(" Temperature sensor 0x%02X temperature readout failed: 0x%04X" % (addr_temp_sens, status))
    return temp

def getCoreTemp():
    return Grimsel.daq.getDouble(f'/{Grimsel.devid}/STATS/PHYSICAL/FPGA/TEMP')

def getPCBRevision(i2c_mux = 1, i2c_addr = MCP23017_addr_frontend):
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', i2c_mux)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', i2c_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', MCP23017_reg_gpioa)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()
    status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
    if (status & 0x0001):
        bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
        if((bin_temp & 0x000F) >> 0) == 0x00:
            brd_type = "MBRD"
        elif((bin_temp & 0x000F) >> 0) == 0x01:
            brd_type = "RF Upconverter"
        elif((bin_temp & 0x000F) >> 0) == 0x02:
            brd_type = "RF Downconverter"
        elif((bin_temp & 0x000F) >> 0) == 0x03:
            brd_type = "Synthesizer BRD"
        elif((bin_temp & 0x000F) >> 0) == 0x04:
            brd_type = "Auxiliary BRD"
        elif((bin_temp & 0x000F) >> 0) == 0x05:
            brd_type = "Memory BRD"
        else:
            brd_type = "Unknown"

        if((bin_temp & 0xF000) >> 12) == 0x00:
            brd_PCBRevision = "0"
        elif((bin_temp & 0xF000) >> 12) == 0x01:
            brd_PCBRevision = "1"
        elif((bin_temp & 0xF000) >> 12) == 0x02:
            brd_PCBRevision = "2"
        else:
            brd_PCBRevision = "99"

        if((bin_temp & 0x0F00) >> 8) == 0x00:
            brd_AssyRevision = "A"
        elif((bin_temp & 0x0F00) >> 8) == 0x01:
            brd_AssyRevision = "B"
        elif((bin_temp & 0x0F00) >> 8) == 0x02:
            brd_AssyRevision = "C"
        elif((bin_temp & 0x0F00) >> 8) == 0x03:
            brd_AssyRevision = "D"
        elif((bin_temp & 0x0F00) >> 8) == 0x04:
            brd_AssyRevision = "E"
        elif((bin_temp & 0x0F00) >> 8) == 0x05:
            brd_AssyRevision = "F"
        elif((bin_temp & 0x0F00) >> 8) == 0x06:
            brd_AssyRevision = "G"
        else:
            brd_AssyRevision = "Z"

        print(" Board Type is: " + brd_type + " v" + brd_PCBRevision + "." + brd_AssyRevision)
        return bin_temp, brd_type + " v" + brd_PCBRevision + "." + brd_AssyRevision
    else:
        print(" Board Type is: none")
        return 0x0000, "none"
    return

def setRefout(divider=1, mode='lvds_0.5', phase_inv='off', powerdown=1):
    if mode=='lvds_0.5':
        drvmode = 0
    elif mode=='lvds_0.75':
        drvmode = 1
    elif mode=='lvds_1.0':
        drvmode = 2
    elif mode=='lvds_1.25':
        drvmode = 3
    elif mode=='hstl':
        drvmode = 4
    elif mode=='hstl_boost':
        drvmode = 5
    else:
        drvmode = 6

    if phase_inv=='on':
        phase = 2
    else:
        phase = 1

    AD9508_drvmode= (powerdown << 7)|(0x0 << 6)|(phase << 4)|(drvmode << 1)|(0x0 << 0)
#    AD9508_drvmode = 0x10       # 0x1A = HSTL Boost, 0x18 = HSTL, 0x16 = LVDSx1.25, 0x14 = LVDS, 0x12 = LVDSx0.75, 0x10=LVDSx0.5
    AD9508_div = divider-1           # LSB of divider ratio register, Division ratio = x -> 0x0A = /10, 0x10 = /16 etc...
#    AD9508_div1 = 0x00          # MSB of divider ratio register, only lowest two bits!

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', RefOut_mux+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', AD9508_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (0x00 << 8)|(AD9508_reg_ID<<0))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (1 << 8) | (2 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()
    status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
    if (status & 0x0001):
        bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
        #print("Readback value: 0x%02X" %(bin_temp))
        if bin_temp == 0x05:
            print(" AD9508 present")
        else:
            print(" AD9508 NOT present")

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', RefOut_mux+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', AD9508_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (0x00<<16)|(AD9508_reg_out0drvmode<<8)|(AD9508_drvmode<<0))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', RefOut_mux+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', AD9508_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (0x00<<8)|(AD9508_reg_out0drvmode<<0))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (1 << 8) | (2 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()
    status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
    if (status & 0x0001):
        bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
        #print("Readback from AD9508: 0x%02X" %(bin_temp))
    else:
        print(" Readback failed!" %(RefOut_mux))

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', RefOut_mux+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', AD9508_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (0x00<<16)|(AD9508_reg_out0div<<8)|(AD9508_div<<0))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', RefOut_mux+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', AD9508_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (0x00<<8)|(AD9508_reg_out0div<<0))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (1 << 8) | (2 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()
    status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
    if (status & 0x0001):
        bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
        #print("Readback from AD9508: 0x%02X" %(bin_temp))
    else:
        print(" Readback failed!" %(RefOut_mux))
    return

def calADC_RF(chan=1, spectrum='off'):
    RF_Freq = 1.0
    DAC_Freq = 1.01
    DAC_Freq2 = 1.01
    RF_Channel = chan
    post='off'

    bw = 1e3*round(abs(DAC_Freq-DAC_Freq2),3)
    DAC_Freq = (DAC_Freq+DAC_Freq2)/2

    #setFreezeAll()

    print(" Calibrating RF-ADC #%d..."%(RF_Channel))
    disablePrint()
    setDownconverter(channel=RF_Channel, path='rf', cal='on', rf_att=20, bb_gain=0, rf_pre='off', rf_postamp=15)
    setDAC(RF_Channel, DAC_Freq, -2, op='single')
    setPLL(RF_Channel, RF_Freq, 0, 0)
    setFreeze(RF_Channel,0)

    time.sleep(0.5)

    setFreeze(RF_Channel,1)
    if spectrum=='on':
        getSpectrum(chan=RF_Channel, center=0, span=2000, peak='on')
    setDAC(RF_Channel)
    setDownconverter(channel=RF_Channel, cal='off')
    setPLL(RF_Channel)
    enablePrint()
    # if spectrum=='on':
    #     input("Press Enter to continue...")
    print(" ...calibration done")
    return

def calADC_RF_noise(chan=1):
    RF_Freq = 1.0
    DAC_Freq = 1.01
    RF_Channel = chan

    #setFreezeAll()

    print(" Calibrating RF-ADC #%d..."%(RF_Channel))
    disablePrint()
    setDownconverter(channel=RF_Channel, path='rf', cal='on', rf_att=10, bb_gain=0, rf_pre='on')
    setDAC(RF_Channel, DAC_Freq, -2, op='noise')
    setPLL(RF_Channel, RF_Freq, 0, 0)
    setFreeze(RF_Channel,0)

    time.sleep(0.5)

    setFreeze(RF_Channel,1)
    setDAC(RF_Channel)
    setDownconverter(channel=RF_Channel, cal='off')
    setPLL(RF_Channel)
    enablePrint()
    print(" ...calibration done")
    return

def calADC_BB(downconv_chan=7, spectrum='off'):
    DAC_Freq = 1.01
    DAC_Freq2 = 1.01

    bw = 1e3*round(abs(DAC_Freq-DAC_Freq2),3)
    DAC_Freq = (DAC_Freq+DAC_Freq2)/2

    print(" Calibrating BB-ADC #%d..."%(downconv_chan))
    disablePrint()
    setDownconverter(downconv_chan, path='bb', cal='on', bb_gain=0, bb_att='on')
    setDAC(downconv_chan, DAC_Freq, -2, op='single', dist=bw)
    setFreeze(downconv_chan,0)

    time.sleep(0.5)
    if spectrum=='on':
        getSpectrum(chan=downconv_chan, center=0, span=2000, peak='on')
    setFreeze(downconv_chan,1)
    setDAC(downconv_chan)
    setDownconverter(channel=downconv_chan, cal='off')
    setPLL(downconv_chan)
    enablePrint()
    # if spectrum=='on':
    #     input("Press Enter to continue...")
    print(" ...calibration done")
    return

def calADC_BB_noise(downconv_chan=7, spectrum='off'):
    RF_Freq = 1.0
    DAC_Freq = 2.01

    print(" Calibrating BB-ADC #%d..."%(downconv_chan))
    disablePrint()
    setDownconverter(downconv_chan, path='bb', cal='on', rf_att=4, bb_gain=10, bb_filt='on')
    setDAC(downconv_chan, DAC_Freq, -6, op='noise')
    setPLL(downconv_chan, RF_Freq, 0, 0)
    setFreeze(downconv_chan,0)

    time.sleep(0.5)
    if spectrum=='on':
        getSpectrum(chan=downconv_chan, center=0, span=2000, peak='on')
    setFreeze(downconv_chan,1)
    setDAC(downconv_chan)
    setDownconverter(channel=downconv_chan, cal='off')
    setPLL(downconv_chan)
    enablePrint()
    # if spectrum=='on':
    #     input("Press Enter to continue...")
    print(" ...calibration done")
    return

def calADC_AUX(chan, spectrum='off'):
    RF_Freq = 1.0
    DAC_Freq = 1.01
    DAC_Freq2 = 1.01
    RF_Channel = chan

    bw = 1e3*round(abs(DAC_Freq-DAC_Freq2),3)
    DAC_Freq = (DAC_Freq+DAC_Freq2)/2

    #setFreezeAll()

    print(" Calibrating AUX-ADC #%d..."%(RF_Channel))
    disablePrint()
    #setPLL(RF_Channel, RF_Freq, 0, powerdown=0)
    setUpconverter(RF_Channel, RF_Freq, rf_disable=0, path='bb', rf_att=10, bb_gain=10, aux='on', aux_gain=0, rf_tap='on')
    setDAC(RF_Channel, DAC_Freq, -2, op='single', dist=bw)
    setFreeze(RF_Channel,0)

    time.sleep(0.5)

    setFreeze(RF_Channel,1)
    if spectrum=='on':
        getSpectrum(chan=RF_Channel, center=1, span=2000, peak='on')
    setDAC(RF_Channel)
    setUpconverter(channel=RF_Channel)
    #setPLL(RF_Channel)
    enablePrint()
    print(" ...calibration done")
    return

def channel_data_is_complex( input_select ):
    return np.less(input_select,4)

def scope_plot_frequency_domain(center, bw, peak):
    # plot the specturm


    for i in range(4):
        plot_legent_txt.append('input %d' % Grimsel.input_select[i])

    # plot settings
    span_GHz = bw/1e3
    major_tick_MHz = span_GHz*1e3/4
    minor_tick_MHz = major_tick_MHz/5
    major_tick_dB = 20
    minor_tick_dB = 10

    if plt.fignum_exists(Grimsel.fig_num):
        # Figure is still opened
        print("Figure still opened")
        plt.clf()
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
    else:
        # Figure is closed
        plt.rcParams['figure.figsize'] = [15, 10] # size of the plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        Grimsel.fig_num = fig.number

    # plt.ion()
    # plt.show()
    Grimsel.peaks={}
#     for channel in range(4):
#         peaks[channel], _ = find_peaks(Grimsel.input_signal_psd_dB[channel][:N], height=peak_threshold, distance=1000)
    legend_text = []
    for channel in range(4) :
        if Grimsel.channel_enable[channel] :
            plt.plot(Grimsel.f[channel] / 1e9, Grimsel.input_signal_psd_dB[channel], plot_color[channel])
            #plt.plot(Grimsel.input_signal_psd_dB[channel][:N])
            legend_text.append(plot_legent_txt[channel])
            #plt.plot(f / 1e9, fs_sine_psd_dB, '-k')
            if peak != 'off':
                #temp, _ = find_peaks(Grimsel.input_signal_psd_dB[channel][:N], height=peak_threshold, distance=1000)
                #temp2 = temp.tolist() #find_peaks(Grimsel.input_signal_psd_dB[channel][:N], height=peak_threshold, distance=1000)
                Grimsel.peaks[channel], _  = find_peaks(Grimsel.input_signal_psd_dB[channel][:Grimsel.N], height=Grimsel.peak_threshold, distance=1000)
                for i in range(len(Grimsel.peaks[channel])):
                    plt.plot(Grimsel.f[channel][Grimsel.peaks[channel][i]] / 1e9, Grimsel.input_signal_psd_dB[channel][Grimsel.peaks[channel][i]], "x", color = 'orange')
                    plt.text(Grimsel.f[channel][Grimsel.peaks[channel][i]] / 1e9, Grimsel.input_signal_psd_dB[channel][Grimsel.peaks[channel][i]], "%1.2f dBFS at %1.3f GHz" %(Grimsel.input_signal_psd_dB[channel][Grimsel.peaks[channel][i]], Grimsel.f[channel][Grimsel.peaks[channel][i]]/1e9))
    # plot settings
    plt.grid()
    plt.xlabel('f [GHz]')
    plt.ylabel('Amplitude [dBFS]')
    #plt.title('downconverted input signals in the frequency domain')
    #plt.legend(['complex input signal', 'full scale sine wave for comparison']);

    # Grid
    #if first==1:
    major_ticks = np.arange(-2, 2, major_tick_MHz/1000)
    minor_ticks = np.arange(-2, 2, minor_tick_MHz/1000)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    major_ticks = np.arange(-160, 10, major_tick_dB)
    minor_ticks = np.arange(-160, 10, minor_tick_dB)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticklabels(major_ticks)

    # Zoom
    #plt.xlim([-span_GHz/2, span_GHz/2]); # two sided
    plt.xlim([center-span_GHz/2, center+span_GHz/2]) # one sided
    plt.ylim([-120,0])

    ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=0.9)

    #plt.legend(['scope channel 0','scope channel 1','scope channel 2','scope channel 3'])
    plt.legend(legend_text)
    #first=0
    # export to png
    #plt.savefig("test_spectrum_rfdc.png")
    plt.show()
    # plt.draw()
    # plt.pause(0.01)

def scope_plot_SFDR(freq_range, sfdr_vector, peak_vector, pll_power_vector, dac_power_vector, title, legend):
    #csfont = {'fontname':'Akkurat_ZI_Regular'}

    major_tick_MHz = 1000
    minor_tick_MHz = 200
    major_tick_dB = 10
    minor_tick_dB = 2

    plt.rcParams['figure.figsize'] = [15, 10] # size of the plot
    plt.rcParams["font.family"] = "Akkurat_ZI_Regular"
    plt.rcParams['font.size'] = 12

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line_sfdr, = plt.plot(freq_range, sfdr_vector, 'blue', label = legend[0])
    line_peak, = plt.plot(freq_range, peak_vector, 'red', label = legend[1])
    line_pll_power, = plt.plot(freq_range, pll_power_vector, 'orange', label = legend[2])
    line_dac_pwr, = plt.plot(freq_range, dac_power_vector, 'green', label = 'DAC Output Power [dBFS]')
    #line_downconv_att, = plt.plot(freq_range, Downconv_att_array, 'purple', label = 'Downconverter Attenuation')
    plt.grid()
    plt.xlabel('f [GHz]')
    plt.ylabel('Trace Unit (see Legend)')

    major_ticks = np.arange(1, 10, major_tick_MHz/1000)
    minor_ticks = np.arange(1, 10, minor_tick_MHz/1000)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    major_ticks = np.arange(-20, 100, major_tick_dB)
    minor_ticks = np.arange(-20, 100, minor_tick_dB)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    #ax.set_yticklabels(major_ticks)

    ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=0.9)
    plt.xlim([np.amin(freq_range), np.amax(freq_range)]) # one sided
    plt.ylim([-20,80])
    plt.legend(handles=[line_sfdr, line_peak, line_pll_power, line_dac_pwr])#, line_upconv_att, line_downconv_att])
    plt.title(title)

    plt.show()

def scope_start_measurement():
    # start measurement
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/enable', 1)
    Grimsel.daq.getDouble(f'/{Grimsel.devid}/scopes/0/enable')
    """ enable the scope, select SW trigger, trigger the scope """
    scope_enable()
    trig_source_select(8)
    trig_start(1) # only needed when triggering via node\n",

def scope_enable() :
    Grimsel.daq.setInt(f'/{Grimsel.devid}/scopes/0/enable', 1)

def scope_disable() :
    Grimsel.daq.setInt(f'/{Grimsel.devid}/scopes/0/enable', 0)

def trig_start(trigger_count = 1, wait_time_s = 0.01) :
    """ Generate a trigger signal """

    for i in range(trigger_count) :
        Grimsel.daq.setInt(f"/{Grimsel.devid}/RAW/DEBUG/{DEBUG_REG_REG_TRIG_CTRL_START}/VALUE", 1)
        time.sleep(wait_time_s)

def trig_source_select(value) :
    """ Select trigger source """

    Grimsel.daq.setInt(f"/{Grimsel.devid}/RAW/DEBUG/{DEBUG_REG_REG_TRIG_CTRL_SRC_SEL}/VALUE", value)

def scope_read_data():
    # wait until ready, then read out the recorded data

    # needed because the dataserver does not remove the extra header information from
    # the vector data... will be removed as soon as this is fixed in the dataserver
    extra_header_length = 17

    # wait until DMA is done
    for i in range(5):
        print(i)
        if Grimsel.daq.getDouble(f'/{Grimsel.devid}/scopes/0/enable') == 0 :
            print("Measurement ready to read out")
            break
        else :
            time.sleep(0.2)

    #Grimsel.daq.sync()

    # read the recorded data
    vector_data   = [[],[],[],[]]
    Grimsel.recorded_data = [[],[],[],[]]
    vector = None

    for channel in range(4) :
        if Grimsel.channel_enable[channel] :
            print("channel ", channel)

            path = f'/{Grimsel.devid}/scopes/0/channels/{channel}/wave'.lower()
            tmp = Grimsel.daq.get(path, flat=True)
            print(tmp)
            if path in tmp:
                vector_data[channel] = tmp[path][0]['vector'].astype('int32')[extra_header_length:extra_header_length+Grimsel.record_length]
                # convert to complex if the channel has complex data (i.e. real / imaginary samples interleaved)
                if channel_data_is_complex(Grimsel.input_select[channel]) :
                    # change the matrix such that first column is the real part, the second column is the imaginary part
                    Grimsel.recorded_data[channel] = np.reshape(vector_data[channel], (int(len(vector_data[channel])/2), 2 ))
                    # multiply by (1, i)
                    Grimsel.recorded_data[channel] = np.matmul(Grimsel.recorded_data[channel], [[1], [-1j]])
                    # transpose
                    Grimsel.recorded_data[channel] = np.matrix.transpose(Grimsel.recorded_data[channel])[0]
                else :
                    Grimsel.recorded_data[channel] = vector_data[channel]
            else:
                Grimsel.recorded_data[channel] = np.zeros(Grimsel.record_length)
        else :
            Grimsel.recorded_data[channel] = np.zeros(Grimsel.record_length)

    for channel in range(4) :
        if Grimsel.channel_enable[channel] :
            print("length _Grimsel.recorded_data[%d]: " % channel, len(Grimsel.recorded_data[channel]))

def scope_plot_time_domain(start=0, stop=10e-6):


    plt.rcParams['figure.figsize'] = [15, 10] # size of the plot

    plt.ion()
    plt.show()

    # plot the signal in the time domain
    subplot = 0
    for channel in range(4) :
        if Grimsel.channel_enable[channel] :
            plt.subplot(1, 1, subplot + 1)
            subplot += 1
            plt.plot(Grimsel.t * 1e6, Grimsel.recorded_data[channel], '.-' ,color=plot_color[channel])

            # plot settings
            plt.grid(True)
            plt.xlabel('time [us]')
            plt.ylabel('digital value')
            #plt.title('input signals in the time domain')
            #plt.legend([plot_legent_txt[channel]])

            # zoom
            plt.ylim([-2**11,2**11])
            #plt.xlim([0,t[-1] * 1e6]) # zoom full
            plt.xlim([start,stop*1e6]); # zoom to 0.1us

    plt.draw()
    plt.pause(0.01)

def scope_calculate_spectrum():
    # calculate the spectrum

    sfdr_list = []
    Grimsel.peaks = {}

    input_signal_psd = [[],[],[],[]]

    Grimsel.f = [[],[],[],[]]

    for channel in range(4) :
        if Grimsel.channel_enable[channel] :
            Grimsel.n_fft = len(Grimsel.recorded_data[channel])

            if Grimsel.window_fct_name == "hanning":
                window_fct = np.hanning(Grimsel.n_fft)
                dB_scale_factor = 2.0020*2/(Grimsel.n_fft*2**11)      # Hamming = 1.85, Hanning = 2.0, Blackman = 2.80
            elif Grimsel.window_fct_name == "hamming":
                window_fct = np.hamming(Grimsel.n_fft)
                dB_scale_factor = 1.8534*2/(Grimsel.n_fft*2**11)      # Hamming = 1.85, Hanning = 2.0, Blackman = 2.80
            elif Grimsel.window_fct_name == "blackman":
                window_fct = np.blackman(Grimsel.n_fft)
                dB_scale_factor = 2.3833*2/(Grimsel.n_fft*2**11)      # Hamming = 1.85, Hanning = 2.0, Blackman = 2.80
            else:
                window_fct = np.ones(Grimsel.n_fft)
                dB_scale_factor = 2/(Grimsel.n_fft*2**11)      # Hamming = 1.85, Hanning = 2.0, Blackman = 2.80

            if channel_data_is_complex(Grimsel.input_select[channel]) :
                #input_signal = np.asarray(Grimsel.recorded_data[channel])
                input_signal_psd[channel] = abs(np.fft.fft(Grimsel.recorded_data[channel]*window_fct, Grimsel.n_fft) )*dB_scale_factor
                Grimsel.f[channel] = np.asarray(range(int(Grimsel.n_fft))) * Grimsel.Fs / Grimsel.n_fft / 2 - Grimsel.Fs/4

                input_signal_psd[channel] = np.fft.fftshift(input_signal_psd[channel])

            else : # scope data is real
                # FFT
                input_signal_psd[channel] = abs(np.fft.rfft(Grimsel.recorded_data[channel]*window_fct, Grimsel.n_fft) )*dB_scale_factor
                Grimsel.f[channel] = np.asarray(range(len(input_signal_psd[channel]))) * Grimsel.Fs / Grimsel.n_fft
            print('a')
        print(input_signal_psd)
    #fs_sine_psd_dB      = 10 * np.log10(fs_sine_psd * dB_scale_factor);
    Grimsel.input_signal_psd_dB = [[],[],[],[]]
    for channel in range(4) :
        #dB_scale_factor = 200e-9 / Grimsel.n_fft * dB_scale_factor
        Grimsel.input_signal_psd_dB[channel] = 20 * np.log10(np.asarray(input_signal_psd[channel]))
        #Grimsel.input_signal_psd_dB[channel] = 10 * np.log10(input_signal_psd[channel])
    Grimsel.peaks, _ = find_peaks(Grimsel.input_signal_psd_dB[0][:int(Grimsel.N)], height=-80, distance=1000)

    for i in Grimsel.peaks:
        sfdr_list.append(Grimsel.input_signal_psd_dB[0][i])

    if len(Grimsel.peaks) >= 2:
        sfdr_list.sort(reverse=True)
        sfdr = sfdr_list[0]-sfdr_list[1]
        peak = sfdr_list[0]
    else:
        sfdr = 0
        peak = sfdr_list[0]

    #sfdr = peak_max - peak_2ndmax
    return Grimsel.input_signal_psd_dB, sfdr, peak

def setScopeSFDR(chan=1):
    Grimsel.record_length = 2**16       # number of samples (will be rounded to a multiple of 32)
    Grimsel.N = int(Grimsel.record_length/2)

    # round Grimsel.record_length to a multiple of 32
    # because the scope can record only multiples of 32
    Grimsel.record_length = round(Grimsel.record_length / 32) * 32
    # enable scope channels 0 to 3

    # sampling frequency
    Grimsel.n_fft = Grimsel.record_length
    Grimsel.f = np.linspace(0, Grimsel.Fs/2, Grimsel.N, endpoint=True)

    channel = {
    0:  4,
    1:  0,
    2:  5,
    3:  1,
    4:  6,
    5:  2,
    6:  7,
    7:  3
    }

    chan=channel.get(chan)

    Grimsel.input_select = [chan, 0, 0, 0]
    Grimsel.channel_enable = [1, 0, 0, 0]

    # config the scope
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/enable', 0)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/length', Grimsel.record_length)

    for channel in range(4):
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/channels/%d/inputselect/' % channel, Grimsel.input_select[channel])
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/channels/%d/enable/' % channel, Grimsel.channel_enable[channel])

    # generate a time signal
    t = np.array(range(0, Grimsel.record_length)) / Grimsel.Fs

def getSpectrum(chan=1, length=2**18, center=1, span=2000, peak='off'):
    Grimsel.record_length = length       # number of samples (will be rounded to a multiple of 32)
    Grimsel.N = int(Grimsel.record_length/2)

    # round Grimsel.record_length to a multiple of 32
    # because the scope can record only multiples of 32
    Grimsel.record_length = round(Grimsel.record_length / 32) * 32
    # enable scope channels 0 to 3

    # sampling frequency
    Grimsel.n_fft = Grimsel.record_length
    Grimsel.f = np.linspace(0, Grimsel.Fs/2, Grimsel.N, endpoint=True)

    channel = {
    0:  4,
    1:  0,
    2:  5,
    3:  1,
    4:  6,
    5:  2,
    6:  7,
    7:  3
    }

    chan=channel.get(chan)

    Grimsel.input_select = [chan, 0, 0, 0]
    Grimsel.channel_enable = [1, 0, 0, 0]

    # config the scope
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/enable', 0)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/length', Grimsel.record_length)

    for channel in range(4):
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/channels/%d/inputselect/' % channel, Grimsel.input_select[channel])
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/channels/%d/enable/' % channel, Grimsel.channel_enable[channel])

    # generate a time signal
    Grimsel.t = np.array(range(0, Grimsel.record_length)) / Grimsel.Fs

    # call the above functions to record measurement data and to plot the results below
    scope_start_measurement()
    scope_read_data()
    scope_calculate_spectrum()
    scope_plot_frequency_domain(center, span, peak)

def getSpectrumMulti(chan=[0, 1, 2, 3], scope_channels=[1, 1, 1, 1], length=2**18, center=1, span=2000, peak='off'):
    Grimsel.record_length = length       # number of samples (will be rounded to a multiple of 32)
    Grimsel.N = int(Grimsel.record_length/2)

    # round Grimsel.record_length to a multiple of 32
    # because the scope can record only multiples of 32
    Grimsel.record_length = round(Grimsel.record_length / 32) * 32
    # enable scope channels 0 to 3

    # sampling frequency
    Grimsel.n_fft = Grimsel.record_length
    f = np.linspace(0, Grimsel.Fs/2, Grimsel.N, endpoint=True)

    channel = {
    0:  4,
    1:  0,
    2:  5,
    3:  1,
    4:  6,
    5:  2,
    6:  7,
    7:  3
    }

    for i in range(len(chan)):
        chan[i]=channel.get(chan[i])

    Grimsel.input_select = [chan[0], chan[1], chan[2], chan[3]]
    Grimsel.channel_enable = scope_channels

    # config the scope
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/enable', 0)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/length', Grimsel.record_length)

    for channel in range(4):
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/channels/%d/inputselect/' % channel, Grimsel.input_select[channel])
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/channels/%d/enable/' % channel, Grimsel.channel_enable[channel])

    # generate a time signal
    Grimsel.t = np.array(range(0, Grimsel.record_length)) / Grimsel.Fs

    # call the above functions to record measurement data and to plot the results below
    scope_start_measurement()
    scope_read_data()
    scope_calculate_spectrum()
    scope_plot_frequency_domain(center, span, peak)

def getTimedomain(chan=1, length=2**18, start=0, stop=1e-6):

    Grimsel.record_length = round(length/32)*32       # number of samples (will be rounded to a multiple of 32)
    # enable scope channels 0 to 3

    channel = {
    0:  4,
    1:  0,
    2:  5,
    3:  1,
    4:  6,
    5:  2,
    6:  7,
    7:  3
    }

    chan=channel.get(chan)

    Grimsel.input_select = [chan, 0, 0, 0]
    Grimsel.channel_enable = [1, 0, 0, 0]

    # config the scope
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/enable', 0)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/length', Grimsel.record_length)

    for channel in range(4):
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/channels/%d/inputselect/' % channel, Grimsel.input_select[channel])
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/scopes/0/channels/%d/enable/' % channel, Grimsel.channel_enable[channel])

    # generate a time signal
    Grimsel.t = np.array(range(0, Grimsel.record_length)) / Grimsel.Fs

    # call the above functions to record measurement data and to plot the results below
    scope_start_measurement()
    scope_read_data()
    #scope_calculate_spectrum()
    scope_plot_time_domain(start, stop)

def setAUXDAC(channel = 'A', value = 2.048, mux = 13):
    """
    *** DESCRIPTION:
    set auxiliary DAC on MBRD or Up-/Downconverters
    *** ARGUMENTS:
    - channel:    output channel, A...D for MBRD and Downconverter, A...H for Upconverter
    - value:      value in Volts, 0...4.096, or alternatively in bits 0...65535
    - mux:        output of I2C multiplexer, 0 for leftmost Frontend, 7 for rightmost, 13 for MBRD
    *** RETURNS:
    -
    """
    upboards = [0,2,4,6]
    if (channel == 'a') | (channel == 'A'):
        dac_chan = LTC2655_reg_DAC_A
    elif (channel == 'b') | (channel == 'B'):
        dac_chan = LTC2655_reg_DAC_B
    elif (channel == 'c') | (channel == 'C'):
        dac_chan = LTC2655_reg_DAC_C
    elif (channel == 'd') | (channel == 'D'):
        dac_chan = LTC2655_reg_DAC_D
    elif mux in upboards and ((channel == 'e') | (channel == 'E')):
        dac_chan = LTC2655_reg_DAC_E
    elif mux in upboards and ((channel == 'f') | (channel == 'F')):
        dac_chan = LTC2655_reg_DAC_F
    elif mux in upboards and ((channel == 'g') | (channel == 'G')):
        dac_chan = LTC2655_reg_DAC_G
    elif mux in upboards and ((channel == 'h') | (channel == 'H')):
        dac_chan = LTC2655_reg_DAC_H
    else:
        dac_chan = LTC2655_reg_DAC_ALL

    if value >= 5:
        LTC2655_value = int(value)
    else:
        LTC2655_value = int(65536/4.096*value)
    #print("DAC-Value: %05d %0.3fV" %(LTC2655_value, 4.096/65536*LTC2655_value),end='\r')
    #print(LTC2655_value)

    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', mux+1)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', LTC2655_addr)
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (LTC2655_reg_update_n<<20)|(dac_chan<<16)|(LTC2655_value<<0))
    Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
    Grimsel.daq.sync()

def getRails():
    INA226_cur_array = Grimsel.INA226_cur_array
    INA226_volt_array = Grimsel.INA226_cur_array
    INA226_pow_array = Grimsel.INA226_cur_array
    unbalance_12V = 0.0

    if len(INA226_addr) != 0:
        print("Reading Measurements:")
        print("---------------------")
        for z in range(len(INA226_addr)):
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', INA226_mux+1)
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', INA226_addr[z])
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', INA226_reg_cur)
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
            Grimsel.daq.sync()
            status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
            if (status & 0x0001):
                bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
                if z==1:
                    unbalance_12V = (1-INA226_cur_array[0]/float(bin_temp*INA226_cur_lsb[z]))*100
                    INA226_cur_array[z-1] = INA226_cur_array[0]+float(bin_temp*INA226_cur_lsb[z])
                else:
                    INA226_cur_array[z-1] = float(bin_temp*INA226_cur_lsb[z])
                #print("%s Current: %0.3fA"%(INA226_dev_names[z], INA226_cur_array[z]))
            else:
                print("Current Readback @ 0x%02X failed!" %(INA226_addr[z]))

            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', INA226_mux+1)
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', INA226_addr[z])
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', INA226_reg_bus)
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
            Grimsel.daq.sync()
            status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
            if (status & 0x0001):
                bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
                if z==1:
                    INA226_volt_array[z-1] = (INA226_volt_array[0]+float(bin_temp*0.00125))/2
                else:
                    INA226_volt_array[z-1] = float(bin_temp*0.00125)
                #print("%s Voltage: %0.3fV"%(INA226_dev_names[z], INA226_volt_array[z]))
            else:
                print("Voltage Readback @ 0x%02X failed!" %(INA226_addr[z]))

            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', INA226_mux+1)
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', INA226_addr[z])
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', INA226_reg_pow)
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
            Grimsel.daq.sync()
            status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
            if (status & 0x0001):
                bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
                if z==1:
                    INA226_pow_array[z-1] = INA226_pow_array[0]+float(bin_temp*25*INA226_cur_lsb[z])
                else:
                    INA226_pow_array[z-1] = float(bin_temp*25*INA226_cur_lsb[z])
                #print("%s Power: %0.3fW\n"%(INA226_dev_names[z], INA226_pow_array[z]))
            else:
                print("Power Readback @ 0x%02X failed!" %(INA226_addr[z]))

        # INA226_cur_array.pop(1)
        # INA226_volt_array.pop(1)
        # INA226_pow_array.pop(1)
        # print((INA226_cur_array,INA226_pow_array,INA226_volt_array))

        for z in range(len(INA226_pow_array)):
            print("%s Current: %0.3fA"%(INA226_dev_names[z], INA226_cur_array[z]))
            print("%s Voltage: %0.3fV"%(INA226_dev_names[z], INA226_volt_array[z]))
            print("%s Power: %0.3fW"%(INA226_dev_names[z], INA226_pow_array[z]))
            if z == 0:
                print("+12V Port unbalance: %0.3f%%"%(unbalance_12V))
            print("")
        print("...done")
    else:
        print("No INA226 present, aborting...")
    Grimsel.INA226_cur_array = INA226_cur_array
    Grimsel.INA226_volt_array = INA226_cur_array
    Grimsel.INA226_pow_array = INA226_cur_array
    return INA226_volt_array, INA226_cur_array, INA226_pow_array

def initINA226():
    INA226_addr=Grimsel.INA226_addr
    INA226_cal=Grimsel.INA226_cal
    INA226_cur_lsb=Grimsel.INA226_cur_lsb
    print("INA226 population check")
    for x in range(len(INA226_addr)):
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', INA226_mux+1)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', INA226_addr[x])
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', INA226_reg_ID)
        Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (2 << 8) | (1 << 0)) # 2 bytes to read, 1 byte to write
        Grimsel.daq.sync()
        status = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/STATUS')
        if (status & 0x0001):
            bin_temp = Grimsel.daq.getInt(f'/{Grimsel.devid}/RAW/DEBUG/I2C/READDATA')
            #print("INA226 Response: 0x%04X" %(bin_temp));
            if bin_temp != 0x5449:
                print("INA226 @ 0x%02X NOT present, removing from List" %(INA226_addr[x]))
                INA226_addr.pop(x)
                INA226_cal.pop(x)
                INA226_cur_lsb.pop(x)
        else:
            print("Readback @ 0x%02X failed!" %(INA226_addr[x]))

    if len(INA226_addr) != 0:
        print("Writing configuration")
        for y in range(len(INA226_addr)):
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', INA226_mux+1)
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', INA226_addr[y])
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (INA226_reg_config<<16)|(INA226_config<<0))
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
            Grimsel.daq.sync()
        #print("...done")
    else:
        print("No INA226 present, aborting...")

    if len(INA226_addr) != 0:
        print("Writing calibration values")
        for y in range(len(INA226_addr)):
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/MUXCHANNEL', INA226_mux+1)
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/ADDRESS', INA226_addr[y])
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/WRITEDATA', (INA226_reg_cal<<16)|(INA226_cal[y]<<0))
            Grimsel.daq.setDouble(f'/{Grimsel.devid}/RAW/DEBUG/I2C/TRANSFER', (0 << 8) | (3 << 0)) # 2 bytes to read, 1 byte to write
            Grimsel.daq.sync()
        #print("...done")
    else:
        print("No INA226 present, aborting...")
    Grimsel.INA226_addr=INA226_addr
    Grimsel.INA226_cal=INA226_cal
    Grimsel.INA226_cur_lsb=INA226_cur_lsb
    return

def initMBRD():
    rev = getPCBRevision(i2c_mux=0, i2c_addr=MCP23017_addr_mbrd)[0]
    if (rev & 0xF00F) == 0x0000:
        initFP()
        initINA226()
        initFanController()
        #setAUXDAC(channel='A', value=0.0000)
        setAUXDAC(channel='A', value=2.2734)
        setIntRef()
        setRefout(divider=1, mode='lvds_0.5', phase_inv='off', powerdown=1)
        setFans()
        #setFans(case_speed = 255, fpga_speed = 255, psu_speed = 128)
    else:
        print("MBRD revision not supported, canceling initialization...")

def initSynthBRD():
    rev = getPCBRevision(i2c_mux=12, i2c_addr=MCP23017_addr_frontend)[0]
    #if (rev & 0xFF0F) == 0x1103:
    setPLL(channel=0)
    setPLL(channel=2)
    setPLL(channel=4)
    setPLL(channel=6)
    #else:
    #    print("SynthBRD revision not supported, canceling initialization...")

def setRFOut(channel=0, center_freq=1.0, inband_freq=0.0, power=0.0, pll_power=0, enable='off', path='rf', rf_tap='off', aux='off', aux_gain=0.0, aux_att='off', mode='norm', dac_power=-2):
    """
    *** DESCRIPTION:
    set RF output of upconverter
    *** ARGUMENTS:
    - channel:    output channel, 0 = far left, 7 = far right
    - rf_freq:    center frequency, 1-8
    - dac_freq:   in-band frequency, -0.5...0.5 for RF or 0...2 GHz for BB
    - power:      output power in dBm, roughly calibrated!
    - enable:     enable or disable output
    - mode:       DAC mode: normal, noise, twotone
    *** RETURNS:
    -
    """
    #set DAC ouput power (dBFS) to safe level
    DAC_Power = dac_power+power2dacpwr(center_freq)
    if DAC_Power > 0:
        DAC_Power = 0

    if path=='rf':
        dac_freq = 2.0-inband_freq
        Upconv_att = power2att_upconv(power-(DAC_Power+2), center_freq)
    else:
        dac_freq=inband_freq

    if enable=='on':
        print("Enabling RF Output #%d"%(channel))
        setFans(case_speed=384)

        if path == 'rf':
            setPLL(channel=channel, freq=center_freq, power=pll_power, powerdown=0, reset=0)
            setUpconverter(channel=channel, freq=center_freq, rf_disable=0, path='rf', rf_att=Upconv_att, bb_gain=-6, aux=aux, aux_gain=int(aux_gain), aux_att=aux_att, rf_tap=rf_tap)
        else:
            bb_att = 'off'
            # limit output power of BB path to 4 dBm max
            if power >= 4:
                power = 4
            elif power <= -35:
                power = -35
            if power <= -16:
                bb_att='on'
                power += 20

            bbgain = int(power+15)
            setUpconverter(channel=channel, freq=center_freq, rf_disable=0, path='bb', rf_att=20, bb_gain=bbgain, bb_att=bb_att, aux=aux, aux_gain=int(aux_gain), rf_tap=rf_tap)

        if mode == 'twotone':
            setDAC(channel, dac_freq, DAC_Power, op='dual', dist=10)
        elif mode == 'noise':
            setDAC(channel, dac_freq, DAC_Power, op='noise')
        else:
            setDAC(channel, dac_freq, DAC_Power)

    else:
        print("Disabling RF Output #%d"%(channel))
        setDAC(channel=channel)
        setUpconverter(channel=channel)
        setPLL(channel=channel)
        setFans()

    # def load_obj(name):
    #     with open(f'{Grimsel.devid}_{name}.pkl', 'rb') as f:
    #         return pickle.load(f)
    #
    #
    # def calibRFdownselector(self, a_channel, a_range, a_freq, a_file=None):
    #     if a_file=='':
    #         CALIBranges=list(range(-30,11,5))
    #         # CALIBranges=list(range(-30,11,5))
    #         CALIBRFfreqs=list(np.array(list(range(10,91,4)))/10)
    #         CALIBRFdown={ch:{freq:{ranges:{'att':0,'pre':True,'post':15} for  ranges in self.CALIBranges} for freq in self.CALIBRFfreqs} for ch in range(8)}
    #         CALIBRFdown['calibrated']=False
    #         # CALIBBBdown={ch:{rang:{'filt':True,'lmh':0,'att': True} for rang in self.CALIBranges} for ch in range(8)}
    #         # CALIBBBdown['calibrated']=False
    #         att=[30,25,20,15,27.5,22.5,16.5,11.5,6.5]
    #         pre=[False]*4+[True]*5
    #         post=[15]*4+[10]*4+[5]*1
    #         ranges=range(-30,11,5)
    #         for ch in range(8):
    #             for idx,ranger in enumerate(list(ranges)[::-1]):
    #                 for freq in self.CALIBRFfreqs:
    #                     CALIBRFdown[ch][freq][ranger]={'att':att[idx], 'pre':pre[idx], 'post':post[idx]}
    #     else:
    #         CALIBRFdown=load_obj(a_file)
    #         CALIBRFfreqs=CALIBRFdown[0].keys()
    #         CALIBranges=CALIBRFdown[0][CALIBRFfreqs[0]].keys()
    #
    #     ranger=next((v for v in CALIBranges if v >= a_range),10)
    #     freq=next((v for v in CALIBRFfreqs if v >= a_freq),9)
    #     return CALIBRFdown[a_channel][freq][ranger]
