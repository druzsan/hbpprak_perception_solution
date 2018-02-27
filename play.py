# Imported Python Transfer Function
@nrp.MapRobotPublisher("eye_tilt", Topic("/robot/eye_tilt/pos", std_msgs.msg.Float64))
@nrp.MapVariable("state", initial_value=("initialized", 0.0), scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def play(t, eye_tilt, state):
    #clientLogger.info("Player: state = " + str(state.value) + ", time = " + str(t))
    if t < 8.0:
        eye_tilt.send_message(std_msgs.msg.Float64(0.0))
        return
    elif t < 10.0:
        eye_tilt.send_message(std_msgs.msg.Float64(-1.0*np.pi/4.0))
        return
    elif state.value[0] == "initialized":
        from rospy import ServiceProxy, wait_for_service, ServiceException
        from std_srvs.srv import Trigger
        wait_for_service("/thimblerigger/start_challenge")
        service = ServiceProxy("/thimblerigger/start_challenge", Trigger)
        try:
            responce = service()
            if responce.success:
                state.value = ("challenge_started", t)
        except ServiceException as exc:
            clientLogger.info("Service did not process request: " + str(exc))
    elif state.value[0] == "challenge_started":
        from rospy import ServiceProxy, wait_for_service, ServiceException
        from std_srvs.srv import Trigger
        wait_for_service("/thimblerigger/step_challenge")
        service = ServiceProxy("/thimblerigger/step_challenge", Trigger)
        try:
            responce = service()
            if responce.success:
                state.value = ("showing_ball", t)
        except ServiceException as exc:
            clientLogger.info("Service did not process request: " + str(exc))
    elif state.value[0] == "ball_recognized" and t - state.value[1] > 2.5:
        from rospy import ServiceProxy, wait_for_service, ServiceException
        from std_srvs.srv import Trigger
        wait_for_service("/thimblerigger/step_challenge")
        service = ServiceProxy("/thimblerigger/step_challenge", Trigger)
        try:
            responce = service()
            if responce.success:
                state.value = ("hiding_ball", t)
        except ServiceException as exc:
            clientLogger.info("Service did not process request: " + str(exc))
    elif state.value[0] == "mugs_recognized" and t - state.value[1] > 2.5:
        from rospy import ServiceProxy, wait_for_service, ServiceException
        from std_srvs.srv import Trigger
        wait_for_service("/thimblerigger/step_challenge")
        service = ServiceProxy("/thimblerigger/step_challenge", Trigger)
        try:
            responce = service()
            if responce.success:
                state.value = ("shuffle", t)
        except ServiceException as exc:
            clientLogger.info("Service did not process request: " + str(exc))
    elif state.value[0] == "ball_predicted" and t - state.value[1] > 2.5:
        from rospy import ServiceProxy, wait_for_service, ServiceException
        from std_srvs.srv import Trigger
        wait_for_service("/thimblerigger/step_challenge")
        service = ServiceProxy("/thimblerigger/step_challenge", Trigger)
        try:
            responce = service()
            if responce.success:
                state.value = ("exit", t)
        except ServiceException as exc:
            clientLogger.info("Service did not process request: " + str(exc))
