from gpiozero import PWMOutputDevice
import time
import pigpio
controller=pigpio.pi()
def gpio_ultrasonicPWM(PWMWave,on):
	if on:
		#controller=pigpio.pi()
		#if not controller.connected:
			#print("Failed to connect")
		controller.hardware_PWM(18, 8000, 500000)
		return controller
	else:
		PWMWave.hardware_PWM(18, 0, 0)
		#PWMWave.stop()
		return None

def gpio_ledPWM(ledPWM, on):
	if on:
		signal=PWMOutputDevice(12)
		signal.frequency=1
		signal.value=.5
		return signal
	else:
		ledPWM.close()
		return None

def main():

	PWMSignal=gpio_ultrasonicPWM(None,True)
	LedPWM=gpio_ledPWM(None,True)
	try:
		for i in range(600000):
			controller.hardware_PWM(18,8000,500000)
			
		gpio_ultrasonicPWM(PWMSignal,False)
		gpio_ledPWM(LedPWM,False)
			
	except:
		gpio_ultrasonicPWM(PWMSignal,False)
		gpio_ledPWM(LedPWM,False)

if __name__ == "__main__":
	main()
