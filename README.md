# COSI-159A: Deep Learning Practice

COSI-159A: Computer Vision, 2023 Spring  

Steps to using this API:
	1. set up python environment with pytorch and TensorFlow installed. To do this, you can use Anaconda to help you do it or press ctrl-shift-p in VSCode and type in 'python: create environment'. Select venv or conda to your liking and continue
	2. main.py accepts args, and here's what they mean:
		--epochs	default:10		epochs each run
		--lr		def:0.1			learning rate, from my testing I found a larger number is good for the first few runs, but if you want to fine tune it, give it a lower value so the loss value goes down. You'll hit a bottleneck if you keep running with the same rate.
		--bs		def:64			batch size
		--save_dir	def:'./save'	saving directory
		--load_dir	def:'./save/mnist.pth'	loading directory
		