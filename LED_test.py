import time
import pigpio

LED_GPIO = 12
LED_FREQ = 1
LED_DUTY = 128

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio not connected")

pi.set_mode(LED_GPIO, pigpio.OUTPUT)
pi.set_PWM_frequency(LED_GPIO, LED_FREQ)
pi.set_PWM_range(LED_GPIO, 255)

for _ in range(3):
    pi.set_PWM_dutycycle(LED_GPIO, LED_DUTY)
    time.sleep(0.3)
    pi.set_PWM_dutycycle(LED_GPIO, 0)
    time.sleep(0.3)

pi.stop()