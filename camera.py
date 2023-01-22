import numpy as np
import cv2
from PIL import Image

import numpy as np
import cv2
from tensorflow.keras.models import model_from_json,load_model
from deepface import DeepFace

import datetime
from threading import Thread
import time
import pandas as pd

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ds_factor=0.6


global last_frame1  

last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)

global cap1 
show_text=[0]


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    	
		def __init__(self, src=0):
			# self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			self.stream = cv2.VideoCapture(src)
			(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
	
	def get_frame(self):
		try:
			global cap1
			global df1
			global jpeg
			# cap1 = WebcamVideoStream(src=0).start()
			cap1 = cv2.VideoCapture(0)
			_,image = cap1.read()
			image=cv2.resize(image,(600,500))
			gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			face_rects=face_cascade.detectMultiScale(gray,1.3,5)
			df1 = music_rec('nutral')
			for (x,y,w,h) in face_rects:
				result = DeepFace.analyze(image,actions = ['emotion'],enforce_detection=False)
				text = result['dominant_emotion']
				cv2.putText(image, text, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
				cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),2)
				df1 = music_rec(text)

			# global last_frame1
			last_frame1 = image.copy()   
			img = Image.fromarray(last_frame1)
			img = np.array(img)
			ret, jpeg = cv2.imencode('.jpg', img)
			return jpeg.tobytes(), df1
		except Exception as e:
			print(str(e))
			return jpeg.tobytes(), df1
		


def music_rec(emotion):

	df_spotify = pd.read_csv('spotify_music_mood.csv')
	df = pd.DataFrame()

	# Happy --> happy and enegrgitic
	'''
		If the mood is happy, it would make sense to recommend music that matches that feeling. 
		Music that is labeled as "happy" or "energetic" would have a positive and upbeat tone, 
		which would enhance and complement the happy mood.
		It can also help to uplift the mood if it's not completely happy.
	'''
	if emotion == 'happy':
		df = df_spotify[df_spotify['mood'].isin(['Happy','Energetic'])]
		select_random_songs = np.random.randint(1,df.shape[0],20)
		df = df.iloc[select_random_songs][['track_name','genre','artist_name','mood']]



	# Sad --> sad and calm
	'''
		These types of music can help to reflect and process the sad feelings and provide a sense of comfort.
	'''
	if emotion == 'sad':
		df = df_spotify[df_spotify['mood'].isin(['Sad','Calm'])]
		select_random_songs = np.random.randint(1,df.shape[0],20)
		df = df.iloc[select_random_songs][['track_name','genre','artist_name','mood']]

	# Surprise --> happy and calm
	'''In case of pleasant surprise, you could recommend music that has the feature "Energetic" or "Happy" to match the excitement and positive feelings. 
	However, in case of shock or unpleasent surprise, you could recommend music that has the feature "Calm" to help the person process and cope with the surprise.
	'''
	if emotion == 'surprise':
		df = df_spotify[df_spotify['mood'].isin(['Happy','Calm'])]
		select_random_songs = np.random.randint(1,df.shape[0],20)
		df = df.iloc[select_random_songs][['track_name','genre','artist_name','mood']]


	# Angry --> Energitic
	'''
	If the mood is angry, you could recommend music that has the feature "Energetic" as it can be a way to release the pent-up energy and frustration caused by the anger
	'''
	if emotion == 'angry':
		df = df_spotify[df_spotify['mood'].isin(['Energetic'])]
		select_random_songs = np.random.randint(1,df.shape[0],20)
		df = df.iloc[select_random_songs][['track_name','genre','artist_name','mood']]

	# Fear --> Calm
	'''
	If the mood is fear, you could recommend music that has the feature "Calm" to help the person process and cope with the fear
	'''
	if emotion == 'fear':
		df = df_spotify[df_spotify['mood'].isin(['Calm'])]
		select_random_songs = np.random.randint(1,df.shape[0],20)
		df = df.iloc[select_random_songs][['track_name','genre','artist_name','mood']]

	# Disgust -- 'Happy', 'Calm'
	if emotion == 'disgust':
		df = df_spotify[df_spotify['mood'].isin(['Happy','Calm'])]
		select_random_songs = np.random.randint(1,df.shape[0],20)
		df = df.iloc[select_random_songs][['track_name','genre','artist_name','mood']]

	# Neutral -- 'Energetic', 'Happy'
	if emotion == 'neutral':
		df = df_spotify[df_spotify['mood'].isin(['Happy','Energetic'])]
		select_random_songs = np.random.randint(1,df.shape[0],20)
		df = df.iloc[select_random_songs][['track_name','genre','artist_name','mood']]

	return df
	
# while True:
# 	VideoCamera().get_frame()
# 	key = cv2.waitKey(1) & 0xff
# 	if key == 27:
# 		break
