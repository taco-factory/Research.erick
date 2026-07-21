#-----------------------------------------------------------------------------#
#-----------------------Quick Start Guide for Python--------------------------#
#-----------------------------------------------------------------------------#
#------------------QBot Platform with Mobile Robotics Lab----------------------#
#-----------------------------------------------------------------------------#

import os
import sys
import platform
import cv2
import numpy as np
import time
from quanser.devices import (
    RangingMeasurements,
    RangingMeasurementMode,
    DeviceError,
    RangingDistance
)
from quanser.multimedia import Video3D, Video3DStreamType, VideoCapture, \
    MediaError, ImageFormat, ImageDataType, VideoCapturePropertyCode, \
    VideoCaptureAttribute
from quanser.communications import Stream, StreamError, PollFlag
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout


class BasicStream:
    '''Class object consisting of basic stream server/client functionality'''
    def __init__(self, uri, agent='S', receiveBuffer=np.zeros(1, dtype=np.float64), sendBufferSize=2048, recvBufferSize=2048, nonBlocking=False, verbose=False):
        self.agent 			= agent
        self.sendBufferSize = sendBufferSize
        self.recvBufferSize = recvBufferSize
        self.uri 			= uri
        self.receiveBuffer  = receiveBuffer
        self.verbose        = verbose

        self.clientStream = Stream()
        if agent=='S':
            self.serverStream = Stream()

        self.t_out = Timeout(seconds=0, nanoseconds=10000000)
        self.connected = False

        try:
            if agent == 'C':
                self.connected = self.clientStream.connect(uri, nonBlocking, self.sendBufferSize, self.recvBufferSize)
                if self.connected and self.verbose:
                    print('Connected to a Server successfully.')

            elif agent == 'S':
                if self.verbose:
                    print('Listening for incoming connections.')
                self.serverStream.listen(self.uri, nonBlocking)

        except StreamError as e:
            if self.agent == 'S' and self.verbose:
                print('Server initialization failed.')
            elif self.agent == 'C' and self.verbose:
                print('Client initialization failed.')
            print(e.get_error_message())

    def checkConnection(self, timeout=Timeout(seconds=0, nanoseconds=100)):
        if self.agent == 'C' and not self.connected:
            try:
                pollResult = self.clientStream.poll(timeout, PollFlag.CONNECT)

                if (pollResult & PollFlag.CONNECT) == PollFlag.CONNECT:
                    self.connected = True
                    if self.verbose: print('Connected to a Server successfully.')

            except StreamError as e:
                if e.error_code == -33:
                    self.connected = self.clientStream.connect(self.uri, True, self.sendBufferSize, self.recvBufferSize)
                else:
                    if self.verbose: print('Client initialization failed.')
                    print(e.get_error_message())

        if self.agent == 'S' and not self.connected:
            try:
                pollResult = self.serverStream.poll(self.t_out, PollFlag.ACCEPT)
                if (pollResult & PollFlag.ACCEPT) == PollFlag.ACCEPT:
                    self.connected = True
                    if self.verbose: print('Found a Client successfully...')
                    self.clientStream = self.serverStream.accept(self.sendBufferSize, self.recvBufferSize)

            except StreamError as e:
                if self.verbose: print('Server initialization failed...')
                print(e.get_error_message())

    def terminate(self):
        if self.connected:
            self.clientStream.shutdown()
            self.clientStream.close()
            if self.verbose: print('Successfully terminated clients...')

        if self.agent == 'S':
            self.serverStream.shutdown()
            self.serverStream.close()
            if self.verbose: print('Successfully terminated servers...')

    def receive(self, iterations=1, timeout=Timeout(seconds=0, nanoseconds=10)):
        self.t_out = timeout
        counter = 0
        dataShape = self.receiveBuffer.shape

        numBytesBasedOnType = len(np.array([0], dtype=self.receiveBuffer.dtype).tobytes())

        dim = 1
        for i in range(len(dataShape)):
            dim = dim*dataShape[i]

        totalNumBytes = dim*numBytesBasedOnType
        self.data = bytearray(totalNumBytes)
        self.bytesReceived = 0

        try:
            while True:
                pollResult = self.clientStream.poll(self.t_out, PollFlag.RECEIVE)
                counter += 1
                if not (iterations == 'Inf'):
                    if counter > iterations:
                        break
                if not ((pollResult & PollFlag.RECEIVE) == PollFlag.RECEIVE):
                    continue

                self.bytesReceived = self.clientStream.receive_byte_array(self.data, totalNumBytes)
                break

            self.receiveBuffer = np.reshape(np.frombuffer(self.data, dtype=self.receiveBuffer.dtype), dataShape)

        except StreamError as e:
            print(e.get_error_message())
        finally:
            receiveFlag = self.bytesReceived==1
            return receiveFlag, totalNumBytes*self.bytesReceived

    def send(self, buffer):
        byteArray = buffer.tobytes()
        self.sentFlag = 0

        try:
            self.sentFlag = self.clientStream.send_byte_array(byteArray, len(byteArray))
            self.clientStream.flush()
        except StreamError as e:
            print(e.get_error_message())
            self.sentFlag = -1
        finally:
            return self.sentFlag


class Camera3D():
    def __init__(
            self,
            mode='RGB, Depth',
            frameWidthRGB=1920,
            frameHeightRGB=1080,
            frameRateRGB=30.0,
            frameWidthDepth=1280,
            frameHeightDepth=720,
            frameRateDepth=15.0,
            frameWidthIR=1280,
            frameHeightIR=720,
            frameRateIR=15.0,
            deviceId='0',
            readMode=1,
            focalLengthRGB=np.array([[None], [None]], dtype=np.float64),
            principlePointRGB=np.array([[None], [None]], dtype=np.float64),
            skewRGB=None,
            positionRGB=np.array([[None], [None], [None]], dtype=np.float64),
            orientationRGB=np.array(
                [[None, None, None], [None, None, None], [None, None, None]],
                dtype=np.float64),
            focalLengthDepth=np.array([[None], [None]], dtype=np.float64),
            principlePointDepth=np.array([[None], [None]], dtype=np.float64),
            skewDepth=None,
            positionDepth=np.array([[None], [None], [None]], dtype=np.float64),
            orientationDepth=np.array(
                [[None, None, None], [None, None, None], [None, None, None]],
                dtype=np.float64)
        ):

        self.mode = mode
        self.readMode = readMode
        self.streamIndex = 0

        self.imageBufferRGB = np.zeros(
            (frameHeightRGB, frameWidthRGB, 3),
            dtype=np.uint8
        )
        self.imageBufferDepthPX = np.zeros(
            (frameHeightDepth, frameWidthDepth, 1),
            dtype=np.uint16
        )
        self.imageBufferDepthM = np.zeros(
            (frameHeightDepth, frameWidthDepth, 1),
            dtype=np.float32
        )
        self.imageBufferIRLeft = np.zeros(
            (frameHeightIR, frameWidthIR, 1),
            dtype=np.uint8
        )
        self.imageBufferIRRight = np.zeros(
            (frameHeightIR, frameWidthIR, 1),
            dtype=np.uint8
        )

        self.frameWidthRGB = frameWidthRGB
        self.frameHeightRGB = frameHeightRGB
        self.frameWidthDepth = frameWidthDepth
        self.frameHeightDepth = frameHeightDepth
        self.frameWidthIR = frameWidthIR
        self.frameHeightIR = frameHeightIR

        self.focalLengthRGB = 2*focalLengthRGB
        self.focalLengthRGB[0, 0] = -self.focalLengthRGB[0, 0]
        self.principlePointRGB = principlePointRGB
        self.skewRGB = skewRGB
        self.positionRGB = positionRGB
        self.orientationRGB = orientationRGB

        self.focalLengthDepth = 2*focalLengthDepth
        self.focalLengthDepth[0, 0] = -self.focalLengthDepth[0, 0]
        self.principlePointDepth = principlePointDepth
        self.skewDepth = skewDepth
        self.positionDepth = positionDepth
        self.orientationDepth = orientationDepth

        try:
            self.video3d = Video3D(deviceId)
            self.streamOpened = False
            if 'rgb' in self.mode.lower():
                self.streamRGB = self.video3d.stream_open(
                    Video3DStreamType.COLOR,
                    self.streamIndex,
                    frameRateRGB,
                    frameWidthRGB,
                    frameHeightRGB,
                    ImageFormat.ROW_MAJOR_INTERLEAVED_BGR,
                    ImageDataType.UINT8
                )
                self.streamOpened = True
            if 'depth' in self.mode.lower():
                self.streamDepth = self.video3d.stream_open(
                    Video3DStreamType.DEPTH,
                    self.streamIndex,
                    frameRateDepth,
                    frameWidthDepth,
                    frameHeightDepth,
                    ImageFormat.ROW_MAJOR_GREYSCALE,
                    ImageDataType.UINT16
                )
                self.streamOpened = True
            if 'ir' in self.mode.lower():
                self.streamIRLeft = self.video3d.stream_open(
                    Video3DStreamType.INFRARED,
                    1,
                    frameRateIR,
                    frameWidthIR,
                    frameHeightIR,
                    ImageFormat.ROW_MAJOR_GREYSCALE,
                    ImageDataType.UINT8
                )
                self.streamIRRight = self.video3d.stream_open(
                    Video3DStreamType.INFRARED,
                    2,
                    frameRateIR,
                    frameWidthIR,
                    frameHeightIR,
                    ImageFormat.ROW_MAJOR_GREYSCALE,
                    ImageDataType.UINT8
                )
                self.streamOpened = True
            self.video3d.start_streaming()
        except MediaError as me:
            print(me.get_error_message())

    def terminate(self):
        try:
            self.video3d.stop_streaming()
            if self.streamOpened:
                if 'rgb' in self.mode.lower():
                    self.streamRGB.close()
                if 'depth' in self.mode.lower():
                    self.streamDepth.close()
                if 'ir' in self.mode.lower():
                    self.streamIRLeft.close()
                    self.streamIRRight.close()

            self.video3d.close()

        except MediaError as me:
            print(me.get_error_message())

    def read_RGB(self):
        timestamp = -1
        try:
            frame = self.streamRGB.get_frame()
            while not frame:
                if not self.readMode:
                    break
                frame = self.streamRGB.get_frame()
            if not frame:
                pass
            else:
                frame.get_data(self.imageBufferRGB)
                timestamp = frame.get_timestamp()
                frame.release()
        except KeyboardInterrupt:
            pass
        except MediaError as me:
            print(me.get_error_message())
        finally:
            return timestamp

    def read_depth(self, dataMode='PX'):
        timestamp = -1
        try:
            frame = self.streamDepth.get_frame()
            while not frame:
                if not self.readMode:
                    break
                frame = self.streamDepth.get_frame()
            if not frame:
                pass
            else:
                if dataMode == 'PX':
                    frame.get_data(self.imageBufferDepthPX)
                elif dataMode == 'M':
                    frame.get_meters(self.imageBufferDepthM)
                timestamp = frame.get_timestamp()
                frame.release()
        except KeyboardInterrupt:
            pass
        except MediaError as me:
            print(me.get_error_message())
        finally:
            return timestamp

    def read_IR(self, lens='LR'):
        timestamp = -1
        try:
            if 'l' in lens.lower():
                frame = self.streamIRLeft.get_frame()
                while not frame:
                    if not self.readMode:
                        break
                    frame = self.streamIRLeft.get_frame()
                if not frame:
                    pass
                else:
                    frame.get_data(self.imageBufferIRLeft)
                    timestamp = frame.get_timestamp()
                    frame.release()
            if 'r' in lens.lower():
                frame = self.streamIRRight.get_frame()
                while not frame:
                    if not self.readMode:
                        break
                    frame = self.streamIRRight.get_frame()
                if not frame:
                    pass
                else:
                    frame.get_data(self.imageBufferIRRight)
                    timestamp = frame.get_timestamp()
                    frame.release()
        except KeyboardInterrupt:
            pass
        except MediaError as me:
            print(me.get_error_message())
        finally:
            return timestamp

    def extrinsics_rgb(self):
        transformFromCameraToBody = np.concatenate(
            (np.concatenate((self.orientationRGB, self.positionRGB), axis=1),
                [[0, 0, 0, 1]]),
            axis=0
        )
        return np.linalg.inv(transformFromCameraToBody)

    def intrinsics_rgb(self):
        return np.array(
            [[self.focalLengthRGB[0,0], self.skewRGB,
                    self.principlePointRGB[0,0]],
             [0, self.focalLengthRGB[1,0], self.principlePointRGB[1,0]],
             [0, 0, 1]],
            dtype = np.float64
        )

    def extrinsics_depth(self):
        transformFromCameraToBody = np.concatenate(
            (np.concatenate((self.orientationDepth, self.positionDepth),
                axis=1), [[0, 0, 0, 1]]),
            axis=0
        )
        return np.linalg.inv(transformFromCameraToBody)

    def intrinsics_depth(self):
        return np.array(
            [[self.focalLengthDepth[0,0], self.skewDepth,
                self.principlePointDepth[0,0]],
             [0, self.focalLengthDepth[1,0], self.principlePointDepth[1,0]],
             [0, 0, 1]],
            dtype = np.float64
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.terminate()


class Camera2D():
    def __init__(
            self,
            cameraId="0",
            frameWidth=820,
            frameHeight=410,
            frameRate=30.0,
            focalLength=np.array([[None], [None]], dtype=np.float64),
            principlePoint=np.array([[None], [None]], dtype=np.float64),
            skew=None,
            position=np.array([[None], [None], [None]], dtype=np.float64),
            orientation=np.array(
                [[None,None,None], [None,None,None], [None,None,None]],
                dtype=np.float64),
            imageFormat = 0,
            brightness = None,
            contrast = None,
            gain = None,
            exposure = None,
        ):
        self.url = "video://localhost:"+cameraId

        self.frameWidth = frameWidth
        self.frameHeight = frameHeight

        self.focalLength = 2*focalLength
        self.focalLength[0, 0] = -self.focalLength[0, 0]
        self.principlePoint = principlePoint
        self.skew = skew
        self.position = position
        self.orientation = orientation
        attributes = []

        if imageFormat == 0:
            self.imageFormat = ImageFormat.ROW_MAJOR_INTERLEAVED_BGR
            self.imageData = np.zeros((frameHeight, frameWidth, 3), dtype=np.uint8)
        else:
            self.imageFormat = ImageFormat.ROW_MAJOR_GREYSCALE
            self.imageData = np.zeros((frameHeight, frameWidth), dtype=np.uint8)

        if brightness is not None:
            attributes.append(VideoCaptureAttribute(VideoCapturePropertyCode.BRIGHTNESS, brightness, True))
        if contrast is not None:
            attributes.append(VideoCaptureAttribute(VideoCapturePropertyCode.CONTRAST, contrast, True))
        if gain:
            attributes.append(VideoCaptureAttribute(VideoCapturePropertyCode.GAIN, gain, True, False))
        if exposure is not None:
            attributes.append(VideoCaptureAttribute(VideoCapturePropertyCode.EXPOSURE, exposure, True, False))

        if not attributes:
            attributes = None
            numAttributes = 0
        else:
            numAttributes = len(attributes)

        try:
            self.capture = VideoCapture(
                self.url,
                frameRate,
                frameWidth,
                frameHeight,
                self.imageFormat,
                ImageDataType.UINT8,
                attributes,
                numAttributes
            )
            self.capture.start()
        except MediaError as me:
            print(me.get_error_message())

    def read(self):
        flag = False
        try:
            flag = self.capture.read(self.imageData)
        except MediaError as me:
            print(me.get_error_message())
        except KeyboardInterrupt:
            print('User Interrupted')
        finally:
            return flag

    def reset(self):
        try:
            self.capture.stop()
            self.capture.start()
        except MediaError as me:
            print(me.get_error_message())

    def terminate(self):
        try:
            self.capture.stop()
            self.capture.close()
        except MediaError as me:
            print(me.get_error_message())

    def extrinsics(self):
        transformFromCameraToBody = np.concatenate(
            (np.concatenate((self.orientation, self.position), axis=1),
                [[0, 0, 0, 1]]), axis=0)
        return np.linalg.inv(transformFromCameraToBody)

    def intrinsics(self):
        return np.array(
            [[self.focalLength[0,0], self.skew, self.principlePoint[0,0]],
             [0, self.focalLength[1,0], self.principlePoint[1,0]], [0, 0, 1]],
            dtype = np.float64
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.terminate()


class Lidar():
    def __init__(
            self,
            type='RPLidar',
            numMeasurements=384,
            rangingDistanceMode=2,
            interpolationMode=0,
            interpolationMaxDistance=0,
            interpolationMaxAngle=0
        ):
        self.numMeasurements = numMeasurements
        self.distances = np.zeros((numMeasurements,1), dtype=np.float32)
        self.angles = np.zeros((numMeasurements,1), dtype=np.float32)
        self._measurements = RangingMeasurements(numMeasurements)
        self._rangingDistanceMode = rangingDistanceMode
        self._interpolationMode = interpolationMode
        self._interpolationMaxDistance = interpolationMaxDistance
        self._interpolationMaxAngle = interpolationMaxAngle

        if type.lower() == 'rplidar':
            self.type = 'RPLidar'
            from quanser.devices import RPLIDAR as RPL
            self._lidar = RPL()
            if not hasattr(self, "url"):
                self.url = ("serial-cpu://localhost:2?baud='115200',"
                        "word='8',parity='none',stop='1',flow='none',dsr='on'")
            self._lidar.open(self.url, self._rangingDistanceMode)

        elif type.lower() == 'leishenms10':
            self.type = 'LeishenMS10'
            from quanser.devices import LeishenMS10
            self._lidar = LeishenMS10()
            if not hasattr(self, "url"):
                self.url = ("serial-cpu://localhost:2?baud='460800',"
                        "word='8',parity='none',stop='1',flow='none'")
            self._lidar.open(self.url, samples_per_scan = self.numMeasurements)

        elif type.lower() == 'leishenm10p':
            self.type = 'LeishenM10P'
            from quanser.devices import LeishenM10P
            self._lidar = LeishenM10P()
            if not hasattr(self, "url"):
                self.url = ("serial://localhost:0?baud='512000',"
                        "word='8',parity='none',stop='1',flow='none',device='/dev/lidar'")
            self._lidar.open(self.url, samples_per_scan = self.numMeasurements)

        else:
            return

        try:
            if rangingDistanceMode == 2:
                self._rangingDistanceMode = RangingDistance.LONG
            elif rangingDistanceMode == 1:
                self._rangingDistanceMode = RangingDistance.MEDIUM
            elif rangingDistanceMode == 0:
                self._rangingDistanceMode = RangingDistance.SHORT
            else:
                print('Unsupported Ranging Distance Mode provided.'
                        'Configuring LiDAR in Long Range mode.')
                self._rangingDistanceMode = RangingDistance.LONG

            if interpolationMode == 0:
                self._interpolationMode = RangingMeasurementMode.NORMAL
            elif interpolationMode == 1:
                self._interpolationMode = RangingMeasurementMode.INTERPOLATED
                self._interpolationMaxAngle = interpolationMaxAngle
                self._interpolationMaxDistance = interpolationMaxDistance
            else:
                print('Unsupported Interpolation Mode provided.'
                        'Configuring LiDAR without interpolation.')
                self._interpolationMode = RangingMeasurementMode.NORMAL

        except DeviceError as de:
            if de.error_code == -34:
                pass
            else:
                print(de.get_error_message())

    def read(self):
        flag = False
        try:
            numValues = self._lidar.read(
                self._interpolationMode,
                self._interpolationMaxDistance,
                self._interpolationMaxAngle,
                self._measurements
            )
            if numValues > 0:
                self.distances = np.array(self._measurements.distance)
                self.angles = np.array(self._measurements.heading)
                flag = True
        except DeviceError as de:
            if de.error_code == -34:
                pass
            else:
                print(de.get_error_message())
        finally:
            return flag

    def terminate(self):
        try:
            self._lidar.close()
        except DeviceError as de:
            if de.error_code == -34:
                pass
            else:
                print(de.get_error_message())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.terminate()


class Probe():
    def __init__(self, ip = 'localhost'):
        self.remoteHostIP = ip
        self.agents = dict()
        self.agentType = []
        self.numDisplays = 0
        self.numPlots = 0
        self.numScopes = 0
        self.connected = False

    def add_display(self,
            imageSize = [480,640,3],
            scaling = True,
            scalingFactor = 2,
            name = 'display'
        ):
        self.numDisplays += 1
        _display = RemoteDisplay(ip = self.remoteHostIP,
            id = self.numDisplays,
            imageSize = imageSize,
            scaling = scaling,
            scalingFactor = scalingFactor
            )

        if name == 'display':
            name = 'display_'+str(self.numDisplays)
        self.agents[name] = (_display, 0)
        return True

    def add_plot(self,
            numMeasurements = 1680,
            scaling = True,
            scalingFactor = 2,
            name = 'plot'
        ):
        self.numPlots += 1
        _plot = RemotePlot(ip = self.remoteHostIP,
                           numMeasurements=numMeasurements,
                           id=self.numPlots,
                           scaling = scaling,
                           scalingFactor = scalingFactor)

        if name == 'plot':
            name = 'plot_'+str(self.numDisplays)
        self.agents[name] = (_plot, 1)
        return True

    def add_scope(self,
            numSignals = 1,
            name = 'scope'
        ):
        self.numScopes += 1
        _scope = RemoteScope(numSignals=numSignals, id=self.numScopes, ip=self.remoteHostIP)

        if name == 'scope':
            name = 'scope_'+str(self.numDisplays)
        self.agents[name] = (_scope, 2)
        return True

    def check_connection(self):
        self.connected = True
        for key in self.agents:
            if not self.agents[key][0].connected:
                self.agents[key][0].check_connection()
            self.connected = self.connected and self.agents[key][0].connected
        return self.connected

    def send(self, name,
             imageData=None,
             lidarData=None,
             scopeData=None):
        flag = False
        agentType = self.agents[name][1]
        if agentType == 0:
            if imageData is None:
                print("Image data not provided for a display agent.")
            else:
                flag = self.agents[name][0].send(imageData)
        elif agentType == 1:
            if lidarData is None:
                print("Lidar data not provided for a plot agent.")
            else:
                flag = self.agents[name][0].send(distances=lidarData[0], angles=lidarData[1])
        elif agentType == 2:
            if scopeData is None:
                print("Scope data not provided for a scope agent")
            else:
                flag = self.agents[name][0].send(scopeData[0], data=scopeData[1])

        return flag

    def terminate(self):
        for key in self.agents:
            self.agents[key][0].terminate()


class RemoteDisplay:
    def __init__(
            self,
            ip = 'localhost',
            id = 0,
            imageSize = [480,640,3],
            scaling = True,
            scalingFactor = 2
        ):
        if scaling and scalingFactor < 1:
            scalingFactor == 1

        if (scaling and
            (imageSize[0] % scalingFactor != 0 or
            imageSize[1] % scalingFactor != 0)):
            sys.exit('Select a scaling factor that is a factor of both width and height of image')

        if scaling:
            imageSize[0] = int(imageSize[0]/scalingFactor)
            imageSize[1] = int(imageSize[1]/scalingFactor)
            self.newSize = (imageSize[1], imageSize[0])

        bufferSize = np.prod(imageSize)
        id = np.clip(id,0,80)
        port = 18800+id
        uriAddress  = 'tcpip://' + str(ip) + ':' + str(port)

        self.client = BasicStream(uriAddress, agent='C',
                                  sendBufferSize=bufferSize,
                                  nonBlocking=True)
        self.scaling = scaling
        self.scalingFactor = scalingFactor
        self.connected = self.client.connected
        self.timeout = Timeout(seconds=0, nanoseconds=10000000)

    def check_connection(self):
        self.client.checkConnection(timeout=self.timeout)
        self.connected = self.client.connected

    def send(self,image):
        if self.client.connected:
            if self.scaling:
                image2 = cv2.resize(image, self.newSize)
                sent = self.client.send(image2)
            else:
                sent = self.client.send(image)
            if sent == -1:
                return False
            else:
                return True

    def terminate(self):
        self.client.terminate()


class RemotePlot:
    def __init__(
            self,
            ip = 'localhost',
            id = 1,
            numMeasurements = 1680,
            scaling = True,
            scalingFactor = 4
        ):
        self.scalingFactor = scalingFactor
        self.scaling = scaling
        self.numMeasurements = numMeasurements
        if scaling:
            self.numMeasurements = int(numMeasurements/self.scalingFactor)
        bufferSize = self.numMeasurements * 2 * 4
        port = 18600+id
        uriAddress  = 'tcpip://' + ip + ':'+ str(port)
        self.client = BasicStream(uriAddress, agent='C',
                                  sendBufferSize=bufferSize,
                                  nonBlocking=True)
        self.connected = self.client.connected
        self.timeout = Timeout(seconds=0, nanoseconds=1)

    def check_connection(self):
        self.client.checkConnection(timeout=self.timeout)
        self.connected = self.client.connected

    def send(self, distances = None, angles = None):
        if distances is None or angles is None:
            return False

        if self.client.connected:
            if self.scaling:
                data = np.concatenate((np.reshape(distances[0:-1:self.scalingFactor], (-1, 1)), np.reshape(angles[0:-1:self.scalingFactor], (-1, 1))), axis=1)
            else:
                data = np.concatenate((np.reshape(distances, (-1, 1)), np.reshape(angles, (-1, 1))), axis=1)
            result = self.client.send(data)
            if result == -1:
                return False
            else:
                return True

    def terminate(self):
        self.client.terminate()


class RemoteScope():
    def __init__(
            self,
            numSignals = 1,
            id = 1,
            ip = 'localhost'
        ):
        self.numMeasurements = numSignals
        bufferSize = (self.numMeasurements+1) * 8
        port = 18700+id
        uriAddress  = 'tcpip://' + ip + ':'+ str(port)
        self.client = BasicStream(uriAddress, agent='C',
                                  sendBufferSize=bufferSize,
                                  nonBlocking=True)
        self.connected = self.client.connected
        self.timeout = Timeout(seconds=0, nanoseconds=1000000)

    def check_connection(self):
        self.client.checkConnection(timeout=self.timeout)
        self.connected = self.client.connected

    def send(self, time, data = None):
        if data is None:
            return False

        if self.client.connected:
            timestamp = np.array([time], dtype=np.float64)
            flag = self.client.send(np.concatenate((timestamp, data)))
            if flag == -1:
                return False
            else:
                return True

    def terminate(self):
        self.client.terminate()


IS_PHYSICAL_QBOTPLATFORM = (('nvidia' == os.getlogin())
                            and ('aarch64' == platform.machine()))

class QBotPlatformDriver():
    def __init__(self, mode=1, ip='192.168.2.15') -> None:
        self.wheelPositions = np.zeros((2), dtype = np.float64)
        self.wheelSpeeds    = np.zeros((2), dtype = np.float64)
        self.motorCmd       = np.zeros((2), dtype = np.float64)
        self.accelerometer  = np.zeros((3), dtype = np.float64)
        self.gyroscope      = np.zeros((3), dtype = np.float64)
        self.currents       = np.zeros((2), dtype = np.float64)
        self.battVoltage    = np.zeros((1), dtype = np.float64)
        self.watchdog       = np.zeros((1), dtype = np.float64)

        self.uri = 'tcpip://'+ip+':18888'
        self._timeout = Timeout(seconds=0, nanoseconds=1000000)

        self._handle = BasicStream(uri=self.uri,
                                    agent='C',
                                    receiveBuffer=np.zeros((17),
                                                           dtype=np.float64),
                                    sendBufferSize=2048,
                                    recvBufferSize=2048,
                                    nonBlocking=True)

        self._sendPacket = np.zeros((10), dtype=np.float64)
        self._sendPacket[0] = mode
        self._mode = mode

        self.status_check('', iterations=20)

    def status_check(self, message, iterations=10):
        self._timeout = Timeout(seconds=0, nanoseconds=1000)
        counter = 0
        while not self._handle.connected:
            self._handle.checkConnection(timeout=self._timeout)
            counter += 1
            if self._handle.connected:
                print(message)
                break
            elif counter >= iterations:
                print('Driver error: status check failed.')
                break

    def read_write_std(self,
                       timestamp,
                       arm = 1,
                       commands=np.zeros((2), dtype=np.float64),
                       userLED=False,
                       color=[1, 0, 1],
                       hold = 0):
        new = False
        self._timeout = Timeout(seconds=0, nanoseconds=10000000)

        if userLED:
            self._sendPacket[1] = 1.0
            self._sendPacket[2:5] = np.array([color[0], color[1], color[2]])
        else:
            self._sendPacket[1] = 0.0
            self._sendPacket[2:5] = np.array([0, 0, 0])

        self._sendPacket[5] = arm
        self._sendPacket[6] = hold
        self._sendPacket[7] = commands[0]
        self._sendPacket[8] = commands[1]
        self._sendPacket[9] = timestamp

        if self._handle.connected:
            self._handle.send(self._sendPacket)
            new, bytesReceived = self._handle.receive(timeout=self._timeout, iterations=5)
            if new:
                self.wheelPositions = self._handle.receiveBuffer[0:2]
                self.wheelSpeeds = self._handle.receiveBuffer[2:4]
                self.motorCmd = self._handle.receiveBuffer[4:6]
                self.accelerometer = self._handle.receiveBuffer[6:9]
                self.gyroscope = self._handle.receiveBuffer[9:12]
                self.currents = self._handle.receiveBuffer[12:14]
                self.battVoltage = self._handle.receiveBuffer[14]
                self.watchdog = self._handle.receiveBuffer[15]
                self.timeStampRecv = self._handle.receiveBuffer[16]

        else:
            self.status_check('Reconnected to QBot Platform Driver.')

        return new

    def terminate(self):
        self._handle.terminate()


class QBotPlatformLidar(Lidar):
    def __init__(
        self,
        numMeasurements=1680,
        interpolationMode=0,
        interpolationMaxDistance=0,
        interpolationMaxAngle=0
        ):

        if IS_PHYSICAL_QBOTPLATFORM:
            self.url = ("serial://localhost:0?baud='512000',"
                        "word='8',parity='none',stop='1',flow='none',device='/dev/lidar'")
        else:
            self.url = "tcpip://localhost:18918"

        super().__init__(
            type='leishenm10p',
            numMeasurements=numMeasurements,
            interpolationMode=interpolationMode,
            interpolationMaxDistance=interpolationMaxDistance,
            interpolationMaxAngle=interpolationMaxAngle
        )


class QBotPlatformRealSense(Camera3D):
    def __init__(
            self,
            mode='RGB&DEPTH',
            frameWidthRGB=640,
            frameHeightRGB=480,
            frameRateRGB=30.0,
            frameWidthDepth=640,
            frameHeightDepth=480,
            frameRateDepth=30.0,
            frameWidthIR=640,
            frameHeightIR=480,
            frameRateIR=30.0,
            readMode=0,
            focalLengthRGB=np.array([[None], [None]], dtype=np.float64),
            principlePointRGB=np.array([[None], [None]], dtype=np.float64),
            skewRGB=None,
            positionRGB=np.array([[None], [None], [None]], dtype=np.float64),
            orientationRGB=np.array([[None, None, None], [None, None, None],
                                     [None, None, None]], dtype=np.float64),
            focalLengthDepth=np.array([[None], [None]], dtype=np.float64),
            principlePointDepth=np.array([[None], [None]], dtype=np.float64),
            skewDepth=None,
            positionDepth=np.array([[None], [None], [None]], dtype=np.float64),
            orientationDepth=np.array([[None, None, None], [None, None, None],
                                       [None, None, None]], dtype=np.float64)
        ):

        if IS_PHYSICAL_QBOTPLATFORM:
            deviceId = '0'
        else:
            deviceId = "0@tcpip://localhost:18917"
            frameWidthRGB = 640
            frameHeightRGB = 480
            frameRateRGB = 30
            frameWidthDepth = 640
            frameHeightDepth = 480
            frameRateDepth = 30
            frameWidthIR = 640
            frameHeightIR = 480
            frameRateIR = 30

        super().__init__(
            mode,
            frameWidthRGB,
            frameHeightRGB,
            frameRateRGB,
            frameWidthDepth,
            frameHeightDepth,
            frameRateDepth,
            frameWidthIR,
            frameHeightIR,
            frameRateIR,
            deviceId,
            readMode,
            focalLengthRGB,
            principlePointRGB,
            skewRGB,
            positionRGB,
            orientationRGB,
            focalLengthDepth,
            principlePointDepth,
            skewDepth,
            positionDepth,
            orientationDepth
        )


class QBotPlatformCSICamera(Camera2D):
    def __init__(
            self,
            frameWidth=640,
            frameHeight=400,
            frameRate=60.0,
            focalLength=np.array([[None], [None]], dtype=np.float64),
            principlePoint=np.array([[None], [None]], dtype=np.float64),
            skew=None,
            position=np.array([[None], [None], [None]], dtype=np.float64),
            orientation=np.array(
                [[None,None,None], [None,None,None], [None,None,None]],
                dtype=np.float64),
            brightness = None,
            contrast = None,
            gain = None,
            exposure = None
        ):

        if IS_PHYSICAL_QBOTPLATFORM:
            deviceId = '0'
        else:
            deviceId = "0@tcpip://localhost:18915"
            frameRate = 30.0
        super().__init__(
            cameraId=deviceId,
            frameWidth=frameWidth,
            frameHeight=frameHeight,
            frameRate=frameRate,
            focalLength = focalLength,
            principlePoint = principlePoint,
            skew = skew,
            position=position,
            orientation=orientation,
            imageFormat=1,
            brightness = brightness,
            contrast = contrast,
            gain = gain,
            exposure = exposure
        )


# Section A - Setup
os.system('quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_qbot_platform -d /tmp')
time.sleep(5)
os.system('quarc_run -r -t tcpip://localhost:17000 qbot_platform_driver_physical.rt-linux_qbot_platform  -d /tmp -uri tcpip://localhost:17099')
time.sleep(3)
print('Driver deployed')

ipHost, ipDriver = '192.168.2.87', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 1, True
frameRate, sampleRate = 60.0, 1/60.0
endFlag, counter, counterRS, counterDown, counterLidar = False, 0, 0, 0, 0
simulationTime = 180.0
startTime = time.time()

def elapsed_time():
    return time.time() - startTime

# Units Conversion Constants
METERS_TO_FEET      = 3.28084
FEET_TO_METERS      = 1.0 / METERS_TO_FEET
MPH_TO_MPS          = 0.44704

# QBot Kinematic Constants
WHEEL_RADIUS = 0.0325                        # meters
WHEEL_BASE   = 0.235                         # meters

# Motion & Path Geometry (Star of David with enclosed Area = 3.0 sq ft)
STAR_SIDE_LENGTH_FT = np.sqrt(3.0 / (3.0 * np.sqrt(3.0))) # ~0.760 ft (0.231 m)

FORWARD_SPEED_MPH     = 2.0                  # Updated forward speed in mph
FORWARD_SPEED_MPS     = FORWARD_SPEED_MPH * MPH_TO_MPS # Low-level driver (~0.894 m/s)

TURN_SPEED_RAD_SEC    = 0.50                 # Angular speed in rad/s (doubled)
TURN_CORRECTION_FACTOR= 1.08                 # Understeer compensation multiplier
SETTLE_TIME           = 0.5                  # Settle time between state transitions (seconds)

# Turn angles for Star of David geometry (radians):
OUTER_TIP_ANGLE_RAD   = (2.0 * np.pi / 3.0) * TURN_CORRECTION_FACTOR  # 120° Clockwise
INNER_VALLEY_ANGLE_RAD = (np.pi / 3.0) * TURN_CORRECTION_FACTOR        # 60° Counter-Clockwise


# Sensor Redundancy Validation Module
def validate_and_fuse_distances_ft(enc_dist_ft, rs_depth_ft, lidar_dist_ft):
    valid_sensors_ft = []
    
    rs_valid = (rs_depth_ft > 0.65) and (rs_depth_ft < 16.4) and not np.isnan(rs_depth_ft)
    lidar_valid = (lidar_dist_ft > 0.33) and (lidar_dist_ft < 26.2) and not np.isnan(lidar_dist_ft)
    
    if rs_valid and lidar_valid:
        if abs(rs_depth_ft - lidar_dist_ft) > 1.15:
            rs_valid = False

    if rs_valid: valid_sensors_ft.append(rs_depth_ft)
    if lidar_valid: valid_sensors_ft.append(lidar_dist_ft)

    if len(valid_sensors_ft) > 0:
        fused_dist_ft = np.mean(valid_sensors_ft)
        status_msg = f"Validated ({len(valid_sensors_ft)} ext sensor(s))"
    else:
        fused_dist_ft = enc_dist_ft
        status_msg = "Fallback to Odometry"

    return fused_dist_ft, status_msg, rs_valid, lidar_valid


# State Machine Initialization
state = 'FORWARD'
segments_completed = 0
initial_wheel_pos = None
stop_start_time = None
last_print_time = 0.0

try:
    myQBot       = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam      = QBotPlatformCSICamera(exposure = 10)
    realSenseCam = QBotPlatformRealSense()
    lidar        = QBotPlatformLidar()

    probe = Probe(ip = ipHost)
    probe.add_display(imageSize = [400, 640, 1], scaling = True, scalingFactor= 2, name="Downward Camera Image")
    probe.add_display(imageSize = [480, 640, 3], scaling = True, scalingFactor= 2, name="RealSense RGB Image")
    probe.add_display(imageSize = [480, 640, 1], scaling = True, scalingFactor= 2, name="RealSense Depth Image")
    probe.add_plot(numMeasurements=1680, scaling=True, scalingFactor= 8, name='Leishen Lidar')
    
    startTime = time.time()
    time.sleep(0.5)
    print(f'Connecting... Driving Star of David at {FORWARD_SPEED_MPH:.1f} mph...')

    while noKill and not endFlag:
        loop_start_time = time.time()
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:
            newHIL = myQBot.read_write_std(timestamp = time.time() - startTime,
                                            arm = arm,
                                            commands = commands)
            if newHIL:
                if initial_wheel_pos is None:
                    initial_wheel_pos = np.copy(myQBot.wheelPositions)

                newDownCam = downCam.read()
                newRealSenseDepth = realSenseCam.read_depth(dataMode='M')
                newRealSenseRGB = realSenseCam.read_RGB()
                newLidar = lidar.read()

                # Read raw distances in feet
                camera_center_depth_ft = -1.0
                if newRealSenseDepth:
                    camera_center_depth_ft = float(realSenseCam.imageBufferDepthM[240, 320, 0]) * METERS_TO_FEET

                forward_lidar_dist_ft = -1.0
                if newLidar and len(lidar.distances) > 0:
                    forward_lidar_dist_ft = float(lidar.distances[len(lidar.distances) // 2]) * METERS_TO_FEET

                # Telemetry update
                if newDownCam: counterDown += 1
                if newRealSenseDepth or newRealSenseRGB: counterRS += 1
                if newLidar: counterLidar += 1

                if counterDown % 4 == 0:
                    probe.send(name="Downward Camera Image", imageData=downCam.imageData)
                if counterRS % 4 == 2:
                    probe.send(name="RealSense Depth Image", imageData=cv2.convertScaleAbs(realSenseCam.imageBufferDepthM, alpha=(255.0/3.0)))
                elif counterRS % 4 == 0:
                    probe.send(name="RealSense RGB Image", imageData=realSenseCam.imageBufferRGB)
                if counterLidar % 6 == 0:
                    probe.send(name="Leishen Lidar", lidarData=(lidar.distances, (np.pi/2 - lidar.angles)))

                # Wheel Odometry calculation in feet
                delta_left = myQBot.wheelPositions[0] - initial_wheel_pos[0]
                delta_right = myQBot.wheelPositions[1] - initial_wheel_pos[1]
                enc_distance_m = ((delta_left + delta_right) / 2.0) * WHEEL_RADIUS
                enc_distance_ft = enc_distance_m * METERS_TO_FEET

                # Perform sensor cross-check
                fused_dist_ft, status_msg, rs_ok, lidar_ok = validate_and_fuse_distances_ft(
                    enc_dist_ft=enc_distance_ft,
                    rs_depth_ft=camera_center_depth_ft,
                    lidar_dist_ft=forward_lidar_dist_ft
                )

                # State Machine for Star of David Trajectory
                if state == 'FORWARD':
                    if enc_distance_ft < STAR_SIDE_LENGTH_FT:
                        forSpd = FORWARD_SPEED_MPS    # 2.0 mph equivalent
                        turnSpd = 0.0
                    else:
                        state = 'STOP_BEFORE_TURN'
                        stop_start_time = time.time()
                        forSpd = 0.0
                        turnSpd = 0.0

                elif state == 'STOP_BEFORE_TURN':
                    forSpd = 0.0
                    turnSpd = 0.0
                    if time.time() - stop_start_time >= SETTLE_TIME:
                        state = 'TURN'
                        initial_wheel_pos = np.copy(myQBot.wheelPositions)

                elif state == 'TURN':
                    angle_turned = ((delta_right - delta_left) / 2.0) * (WHEEL_RADIUS / (WHEEL_BASE / 2.0))
                    
                    # Alternating turn logic: Even = Outer Tip (+120°), Odd = Inner Valley (-60°)
                    if segments_completed % 2 == 0:
                        target_angle = OUTER_TIP_ANGLE_RAD
                        turn_dir = 1.0   # Turn Right
                    else:
                        target_angle = INNER_VALLEY_ANGLE_RAD
                        turn_dir = -1.0  # Turn Left

                    if abs(angle_turned) < target_angle:
                        forSpd = 0.0
                        turnSpd = turn_dir * TURN_SPEED_RAD_SEC
                    else:
                        segments_completed += 1
                        if segments_completed >= 12:  # 12 perimeter segments
                            state = 'FINISHED'
                        else:
                            state = 'STOP_BEFORE_FORWARD'
                            stop_start_time = time.time()
                        forSpd = 0.0
                        turnSpd = 0.0

                elif state == 'STOP_BEFORE_FORWARD':
                    forSpd = 0.0
                    turnSpd = 0.0
                    if time.time() - stop_start_time >= SETTLE_TIME:
                        state = 'FORWARD'
                        initial_wheel_pos = np.copy(myQBot.wheelPositions)

                elif state == 'FINISHED':
                    forSpd = 0.0
                    turnSpd = 0.0
                    arm = 0

                commands = np.array([forSpd, turnSpd], dtype=np.float64)

                if time.time() - last_print_time > 0.1:
                    print(f"[{time.strftime('%H:%M:%S')}] Seg: {segments_completed:02d}/12 | State: {state:<18} | Dist: {enc_distance_ft:.2f}/{STAR_SIDE_LENGTH_FT:.2f}ft | Speed: {FORWARD_SPEED_MPH:.1f}mph | Status: {status_msg}")
                    last_print_time = time.time()

            endFlag = (t > simulationTime) or (state == 'FINISHED')

        time_to_sleep = sampleRate - (time.time() - loop_start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

except KeyboardInterrupt:
    print('\n[ALERT] Ctrl+C detected! Halting robot...')

finally:
    try:
        myQBot.read_write_std(timestamp=0, arm=0, commands=np.array([0.0, 0.0], dtype=np.float64))
    except:
        pass

    downCam.terminate()
    myQBot.terminate()
    realSenseCam.terminate()
    lidar.terminate()
    probe.terminate()
    print('Star path completed. All systems safely powered down.')