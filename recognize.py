# Imported Python Transfer Function
import numpy as np
import cv2
from cv_bridge import CvBridge
@nrp.MapVariable("state", initial_value=("initialized", 0.0), scope=nrp.GLOBAL)
@nrp.MapRobotSubscriber("camera", Topic("/icub_model/left_eye_camera/image_raw", sensor_msgs.msg.Image))
@nrp.MapVariable("mug", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable("mugs", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable("lastImg", initial_value=(None, 0), scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def recognize(t, state, camera, mug, mugs, lastImg):
    # Helper functions --------------------------------------------------------------------------------------------
    def isEmpty(array):
        return array is None or array.size == 0
    def isNotEmpty(array):
        return not (array is None) and array.size != 0
    def segmentColor(colorImage, color):
        colorToDetect = cv2.absdiff(colorImage[:, :, 2 if color == 'red' else 1], colorImage[:, :, 0])
        return cv2.threshold(colorToDetect, 63, 255, cv2.THRESH_BINARY)[1]
    def recognizeCircles(binaryImage, minRadius, maxRadius, param2):
        return cv2.HoughCircles(binaryImage, cv2.HOUGH_GRADIENT, 1, 2*minRadius, param1=1,
                                param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    def detectBall(colorImage):
        green = segmentColor(colorImage, 'green')
        circles = recognizeCircles(green, minRadius=4, maxRadius=15, param2=8)
        if isEmpty(circles):
            return None
        return np.uint16(np.around(circles[0, 0]))
    def detectMugs(colorImage):
        red = segmentColor(colorImage, 'red')
        circles = recognizeCircles(red, minRadius=5, maxRadius=20, param2=9)
        if isEmpty(circles):
            return None
        circles = np.uint16(np.around(circles[0]))
        return circles[circles[:, 1].argsort()]
    def detectCorrespondedMug(ball, mugs):
        return mugs[np.argmin(np.sqrt((np.array(mugs[:, 0], dtype=np.float) - np.float(ball[0]))**2 +\
                                      (np.array(mugs[:, 1], dtype=np.float) - np.float(ball[1]))**2))]
    def detectTrackedMug(mug, colorImage):
        radius = mug[2] + 10
        offsetX, offsetY, size = max(mug[0] - radius, 0), max(mug[1] - radius, 0), 2*radius
        croppedImage = colorImage[offsetY:offsetY + size, offsetX:offsetX + size]
        red = segmentColor(croppedImage, 'red')
        circles = recognizeCircles(red, minRadius=mug[2] - 1, maxRadius=mug[2] + 1, param2=4)
        if isEmpty(circles):
            return None
        return np.uint16(np.around(circles[0])) + np.array([offsetX, offsetY, 0])
    def trackMug(mug, colorImage):
        mugs = detectTrackedMug(mug, colorImage)
        return detectCorrespondedMug(mug, mugs) if isNotEmpty(mugs) else None
    def predictMug(mugs, ball):
        return 0 if mugs.shape[0] == 1 else mugs.shape[0] - 1 - np.argmin(np.sqrt(\
            (np.array(mugs[:, 0], dtype=np.float) - np.float(ball[0]))**2 +\
            (np.array(mugs[:, 1], dtype=np.float) - np.float(ball[1]))**2))
    # Helper functions --------------------------------------------------------------------------------------------
    #clientLogger.info("Recognizer: state = " + str(state.value) + ", time = " + str(t))
    if state.value[0] == "showing_ball" and t - state.value[1] > 2.5:
        clientLogger.info("Recognizer: recognizing showed ball")
        if not camera.value:
            return
        img = CvBridge().imgmsg_to_cv2(camera.value, "bgr8")
        currentMug = detectBall(img)
        clientLogger.info("Ball coordinates: " + str(currentMug))
        if isNotEmpty(currentMug):
            mug.value = currentMug
            state.value = ("ball_recognized", t)
    elif state.value[0] == "hiding_ball" and t - state.value[1] > 2.5:
        clientLogger.info("Recognizer: recognizing mugs")
        if not camera.value:
            return
        img = CvBridge().imgmsg_to_cv2(camera.value, "bgr8")
        currentMugs = detectMugs(img)
        clientLogger.info('Initial coordinates of mugs:\n' + str(currentMugs))
        if isEmpty(currentMugs):
            return
        mugs.value = currentMugs
        clientLogger.info("Ball coordinates: " + str(mug.value))
    	currentMug = detectCorrespondedMug(mug.value, currentMugs)
        clientLogger.info('Corresponded mug: ' + str(currentMug))
        if isNotEmpty(currentMug):
            clientLogger.info("The ball is under the " + str(predictMug(currentMugs, currentMug) + 1) + "th mug from me")
            mug.value = currentMug
            lastImg.value = (img, 0)
            state.value = ("mugs_recognized", t)
    elif state.value[0] == "shuffle":
        clientLogger.info("Recognizer: tracking mug with ball")
        if lastImg.value[1] > 50:
            clientLogger.info("Recognizer: shuffling is done")
            state.value = ("challenge_stopped", t)
            return
        if not camera.value:
            return
        img = CvBridge().imgmsg_to_cv2(camera.value, "bgr8")
        currentMug = trackMug(mug.value, img)
        clientLogger.info('Corresponded mug: ' + str(currentMug))
        if isNotEmpty(currentMug):
            mug.value = currentMug
        lastImg.value = (lastImg.value[0], lastImg.value[1] + 1)\
            if np.sum(cv2.absdiff(img, lastImg.value[0])) < 480000 else (img, 0)
    elif state.value[0] == "challenge_stopped":
        clientLogger.info("Recognizer: prediction ball position")
        clientLogger.info("I think the ball is under the " + str(predictMug(mugs.value, mug.value) + 1) + "th mug from me")
        state.value = ("ball_predicted", t)
