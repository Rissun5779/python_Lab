import time
import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = "/dev/ttyUSB0"
BAUD = 115200

def run_all():
    tests = [
        (3.4, False, 1000, 880),   # 5 Hz: 1 kHz 取樣, 600 點 (0.6秒 = 3週期)
        (17.0, False, 250, 940),   # 25 Hz: 4 kHz 取樣, 640 點 (0.16秒 = 4週期)
        (170.0, False, 100, 580),  # 250 Hz: 10 kHz 取樣, 400 點 (0.04秒 = 10週期)
        (6.8, True, 500, 880),    # 10 Hz 方波: 2 kHz 取樣, 600 點 (0.3秒 = 3週期)
    ]

    with serial.Serial(PORT, BAUD, timeout=2) as ser:
        time.sleep(1.5)  # 等待開啟序列埠造成的 ESP32 自動重啟完成
        ser.reset_input_buffer()

        for freq, is_sq, sample_us, n_samples in tests:
            cmd = f"{freq},{1 if is_sq else 0},{sample_us},{n_samples}\n"
            ser.write(cmd.encode())

            t_us, vin, vout = [], [], []
            recording = False
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line == "START":
                    recording = True
                    continue
                if line == "END":
                    break
                if recording and ',' in line:
                    p = line.split(',')
                    t_us.append(float(p[0]))
                    vin.append(float(p[1]) * 3.3 / 4095.0)
                    vout.append(float(p[2]) * 3.3 / 4095.0)

            t = (np.array(t_us) - t_us[0]) * 1e-6
            vin, vout = np.array(vin), np.array(vout)

            # 均勻插值至 20 kHz 以精準計算高頻時間差與相位差
            t_u = np.linspace(t[0], t[-1], int((t[-1] - t[0]) * 20000))
            vin_u = np.interp(t_u, t, vin)
            vout_u = np.interp(t_u, t, vout)

            vpp_in = np.max(vin_u) - np.min(vin_u)
            vpp_out = np.max(vout_u) - np.min(vout_u)
            ratio = vpp_out / vpp_in

            vin_ac, vout_ac = vin_u - np.mean(vin_u), vout_u - np.mean(vout_u)
            corr = np.correlate(vout_ac, vin_ac, mode='full')
            lag = np.argmax(corr) - (len(vin_ac) - 1)
            dt = lag * (t_u[1] - t_u[0])
            phase_deg = -360.0 * freq * dt

            wave_name = "Square" if is_sq else "Sine"
            print(f"[{freq:5.1f} Hz {wave_name:6s}] Vin_pp={vpp_in:.3f}V | Vout_pp={vpp_out:.3f}V | 振幅比={ratio:.4f} | 相位差={phase_deg:.2f}°")

            plt.figure(figsize=(9, 3.8))
            plt.plot(t * 1000, vin, label='CH1: Vin (GPIO34)', alpha=0.85)
            plt.plot(t * 1000, vout, label='CH2: Vout (GPIO35)', linewidth=2)
            plt.title(f'ESP32 Scope - {freq} Hz {wave_name} (Ratio={ratio:.3f}, Phase={phase_deg:.1f}°)')
            plt.xlabel('Time (ms)')
            plt.ylabel('Voltage (V)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"scope_{int(freq)}Hz_{wave_name}.png", dpi=150)
            plt.show()

if __name__ == "__main__":
    run_all()