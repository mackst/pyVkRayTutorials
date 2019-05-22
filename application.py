import os
import inspect
from vulkan import *
from PySide2 import (QtGui, QtCore)
from PIL import Image
import numpy as np


__currentDir__ = os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename))


class InstanceProcAddr(object):
    T = None

    def __init__(self, func):
        self.__func = func

    def __call__(self, *args, **kwargs):
        funcName = self.__func.__name__
        func = InstanceProcAddr.procfunc(funcName)
        if func:
            return func(*args, **kwargs)
        else:
            return VK_ERROR_EXTENSION_NOT_PRESENT

    @staticmethod
    def procfunc(funcName):
        return vkGetInstanceProcAddr(InstanceProcAddr.T, funcName)


class DeviceProcAddr(InstanceProcAddr):

    @staticmethod
    def procfunc(funcName):
        return vkGetDeviceProcAddr(DeviceProcAddr.T, funcName)


class Win32misc(object):
    @staticmethod
    def getInstance(hWnd):
        from cffi import FFI as _FFI
        _ffi = _FFI()
        _ffi.cdef('long __stdcall GetWindowLongA(void* hWnd, int nIndex);')
        _lib = _ffi.dlopen('User32.dll')
        return _lib.GetWindowLongA(_ffi.cast('void*', hWnd), -6)  # GWL_HINSTANCE


class Settings(object):
    validationEnabled = True
    desiredWinWidth = 1280
    desiredWinHeight = 720
    desiredSurfaceFormat = VK_FORMAT_B8G8R8A8_UNORM
    # desiredSurfaceFormat = VK_FORMAT_B8G8R8_UNORM


class QueueInfo(object):

    def __init__(self):
        self.queueFamilyIndex = -1
        self.queue = None


class QueuesInfo(object):

    def __init__(self):
        self.graphics = QueueInfo()
        self.compute = QueueInfo()
        self.transfer = QueueInfo()


class ResourceBase(object):
    physicalDevice = None
    device = None
    physicalDeviceMemoryProperties = None
    commandPool = None
    transferQueue = None

    @staticmethod
    def init(physicalDevice, device, commandPool, transferQueue):
        ResourceBase.physicalDevice = physicalDevice
        ResourceBase.device = device
        ResourceBase.physicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties(physicalDevice)
        ResourceBase.commandPool = commandPool
        ResourceBase.transferQueue = transferQueue

    @staticmethod
    def getMemoryType(memoryRequiriments, properties):
        for i, prop in enumerate(ResourceBase.physicalDeviceMemoryProperties.memoryTypes):
            if (memoryRequiriments.memoryTypeBits & (1 << i)) and ((prop.propertyFlags & properties) == properties):
                return i
        return -1


class ImageResource(ResourceBase):
    folderPath = ''

    def __init__(self):
        self.image = None
        self.memory = None
        self.imageView = None
        self.sampler = None

    def __del__(self):
        self.cleanup()

    def createImage(self, imageType, imFormat, extent, tiling, usage, memoryProperties):
        createInfo = VkImageCreateInfo(
            imageType=imageType,
            format=imFormat,
            extent=extent,
            mipLevels=1,
            arrayLayers=1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=tiling,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED
        )

        self.image = vkCreateImage(self.device, createInfo, None)

        memoryRequirements = vkGetImageMemoryRequirements(self.device, self.image)
        memAllocateInfo = VkMemoryAllocateInfo(
            allocationSize=memoryRequirements.size,
            memoryTypeIndex=self.getMemoryType(memoryRequirements, memoryProperties)
        )

        self.memory = vkAllocateMemory(self.device, memAllocateInfo, None)
        vkBindImageMemory(self.device, self.image, self.memory, 0)


    def loadTexture2DFromFile(self, texturePath):
        _image = Image.open(texturePath)
        _image.putalpha(1)
        width = _image.width
        height = _image.height
        imageSize = width * height * 4

        stagingBuffer = BufferResource()
        stagingBuffer.create(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        stagingBuffer.copyToBufferUsingMapUnmap(_image.tobytes(), imageSize)

        del _image

        imageExtent = VkExtent3D(width, height, 1)
        self.createImage(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB, imageExtent, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        allocInfo = VkCommandBufferAllocateInfo(
            commandPool=self.commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        cmdBuffer = vkAllocateCommandBuffers(self.device, allocInfo)[0]
        beginInfo = VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vkBeginCommandBuffer(cmdBuffer, beginInfo)

        barrier = VkImageMemoryBarrier(
            srcAccessMask=0,
            dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=self.image,
            subresourceRange=VkImageSubresourceRange(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1)
        )
        vkCmdPipelineBarrier(self.commandPool, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, None, 0, None, 1, barrier)

        region = VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=VkImageSubresourceLayers(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1),
            imageOffset=[0, 0, 0],
            imageExtent=imageExtent
        )
        vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer.buffer, self.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, region)

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             0, None, 0, None, 1, barrier)
        vkEndCommandBuffer(cmdBuffer)

        submitInfo = VkSubmitInfo(
            waitSemaphoreCount=0,
            commandBufferCount=1,
            pCommandBuffers=cmdBuffer,
            signalSemaphoreCount=0
        )
        vkQueueSubmit(self.transferQueue, 1, submitInfo, None)
        vkQueueWaitIdle(self.transferQueue)
        vkFreeCommandBuffers(self.device, self.commandPool, 1, cmdBuffer)

    def createImageView(self, viewType, imformat, subresourceRange):
        info = VkImageViewCreateInfo(
            viewType=viewType,
            format=imformat,
            subresourceRange=subresourceRange,
            image=self.image,
            components=VkComponentMapping(
                VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A
            )
        )
        self.imageView = vkCreateImageView(self.device, info, None)

    def createSampler(self, magFilter, minFilter, mipmapMode, addressMode):
        info = VkSamplerCreateInfo(
            magFilter=magFilter,
            minFilter=minFilter,
            addressModeU=addressMode,
            addressModeV=addressMode,
            addressModeW=addressMode,
            mipLodBias=0,
            anisotropyEnable=VK_FALSE,
            maxAnisotropy=1,
            compareEnable=VK_FALSE,
            compareOp=VK_COMPARE_OP_ALWAYS,
            minLod=0,
            maxLod=0,
            borderColor=VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            unnormalizedCoordinates=VK_FALSE
        )
        self.sampler = vkCreateSampler(self.device, info, None)

    def cleanup(self):
        if self.imageView:
            vkDestroyImageView(self.device, self.imageView, None)
            self.imageView = None
        if self.memory:
            vkFreeMemory(self.device, self.memory, None)
            self.memory = None
        if self.image:
            vkDestroyImage(self.device, self.image, None)
            self.image = None
        if self.sampler:
            vkDestroySampler(self.device, self.sampler, None)
            self.sampler = None


class ShaderResource(ResourceBase):
    folderPath = ''

    def __init__(self):
        self._module = None

    def __del__(self):
        self.cleanup()

    def loadFromFile(self, filename):
        with open(filename, 'rb') as sf:
            code = sf.read()

            createInfo = VkShaderModuleCreateInfo(
                codeSize=len(code),
                pCode=code
            )

            self._module = vkCreateShaderModule(self._device, createInfo, None)

    def getShaderStage(self, stage):
        info = VkPipelineShaderStageCreateInfo(
            stage=stage,
            module=self._module,
            pName='main'
        )
        return info

    def cleanup(self):
        if self._module:
            vkDestroyShaderModule(self.device, self._module, None)
            self._module = None


class BufferResource(ResourceBase):

    def __init__(self):
        self.buffer = None
        self.memory = None
        self.size = 0

    def __del__(self):
        self.cleanup()

    def create(self, size, usage, memoryProperties):
        createInfo = VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0
        )
        self.size = size

        self.buffer = vkCreateBuffer(self.device, createInfo, None)

        memoryRequirements = vkGetBufferMemoryRequirements(self.device, self.buffer)

        memAllocateInfo = VkMemoryAllocateInfo(
            allocationSize=memoryRequirements.size,
            memoryTypeIndex=self.getMemoryType(memoryRequirements, memoryProperties)
        )
        self.memory = vkAllocateMemory(self.device, memAllocateInfo, None)

        vkBindBufferMemory(self.device, self.buffer, self.memory, 0)


    def map(self, size):
        return vkMapMemory(self.device, self.memory, 0, size, 0)

    def unmpa(self):
        vkUnmapMemory(self.device, self.memory)

    def copyToBufferUsingMapUnmap(self, memoryToCopyFrom, size):
        mappedMemory = self.map(size)
        ffi.memmove(mappedMemory, memoryToCopyFrom, size)
        self.unmpa()
        return True

    def cleanup(self):
        if self.buffer:
            vkDestroyBuffer(self.device, self.buffer, None)
            self.buffer = None
        if self.memory:
            vkFreeMemory(self.device, self.memory, None)
            self.memory = None


@InstanceProcAddr
def vkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, pAllocator):
    pass


@InstanceProcAddr
def vkCreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator):
    pass


@InstanceProcAddr
def vkDestroySurfaceKHR(instance, surface, pAllocator):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface):
    pass


# device extention
@DeviceProcAddr
def vkCreateSwapchainKHR(device, pCreateInfo, pAllocator):
    pass


@DeviceProcAddr
def vkDestroySwapchainKHR(device, swapchain, pAllocator):
    pass


@DeviceProcAddr
def vkGetSwapchainImagesKHR(device, swapchain):
    pass


@DeviceProcAddr
def vkAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence):
    pass


@DeviceProcAddr
def vkQueuePresentKHR(queue, pPresentInfo):
    pass


class Application(QtGui.QWindow):

    def __init__(self):
        super(Application, self).__init__()

        self._appName = 'pyVK tutorial'
        self._settings = Settings()

        self._surfaceFormat = None
        self._instance = None
        self._physicalDevice = None
        self._queuesInfo = QueuesInfo()
        self._device = None
        self._debugMessenger = None
        self._surface = None
        self._swapchain = None
        self._swapchainImages = []
        self._swapchainImageViews = []
        self._offsreenImageResource = ImageResource()
        self._commandPool = None
        self._commandBuffers = None
        self._imageAcquiredSemaphore = None
        self._renderFinishedSemaphore = None
        self._frameReadinessFences = []
        self._bufferedFrameMaxNum = 0

    def __del__(self):
        self.shutdonw()
        if self._renderFinishedSemaphore:
            vkDestroySemaphore(self._device, self._renderFinishedSemaphore, None)
        if self._imageAcquiredSemaphore:
            vkDestroySemaphore(self._device, self._imageAcquiredSemaphore, None)
        if self._commandBuffers:
            vkFreeCommandBuffers(self._device, self._commandPool, len(self._commandBuffers), self._commandBuffers)
        if self._commandPool:
            vkDestroyCommandPool(self._device, self._commandPool, None)
        del self._offsreenImageResource

        for imv in self._swapchainImageViews:
            vkDestroyImageView(self._device, imv, None)
        if self._swapchain:
            vkDestroySwapchainKHR(self._device, self._swapchain, None)
        if self._surface:
            vkDestroySurfaceKHR(self._device, self._surface, None)
        if self._debugMessenger:
            vkDestroyDebugUtilsMessengerEXT(self._instance, self._debugMessenger, None)
        if self._device:
            vkDestroyDevice(self._device, None)
        if self._instance:
            vkDestroyInstance(self._instance, None)

    def __debugCallback(self, messageSeverity, messageType, callbackData, userData):
        print(ffi.string(callbackData.pMessage))
        return 0

    def run(self):
        self.initialize()
        self.show()
        self.drawFrame()

    def initialize(self):
        self.initCommon()
        self.createAppWindow()
        self.createInstance()
        self.createDebugMessenger()
        self.findDeviceAndQueues()
        self.createDevice()
        self.postCreateDevice()
        self.createSurface()
        self.createSwapchain()
        self.createFences()
        self.createCommandPool()
        ResourceBase.init(self._physicalDevice, self._device, self._commandPool, self._queuesInfo.graphics.queue)
        self.createOffsreenBuffers()
        self.createCommandBuffers()
        self.createSynchronization()

        self.init()

        self.fillCommandBuffers()


    def shutdonw(self):
        vkDeviceWaitIdle(self._device)

    def initCommon(self):
        ShaderResource.folderPath = os.path.join(__currentDir__, 'Assets', 'Shaders')
        ImageResource.folderPath = os.path.join(__currentDir__, 'Assets', 'Textures')

    def createAppWindow(self):
        self.setWidth(self._settings.desiredWinWidth)
        self.setHeight(self._settings.desiredWinHeight)
        self.setTitle(self._appName)

    def createInstance(self):
        appInfo = VkApplicationInfo(
            # sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self._appName,
            applicationVersion=1,
            pEngineName='pyvulkan',
            engineVersion=1,
            apiVersion=VK_MAKE_VERSION(1, 1, 0)
        )

        # extenstions = [
        #     VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        # ]
        extenstions = [e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)]
        enabledLayers = []
        if self._settings.validationEnabled:
            enabledLayers.append('VK_LAYER_KHRONOS_validation')
            if 'VK_EXT_debug_utils' not in extenstions:
                extenstions.append('VK_EXT_debug_utils')
        instanceInfo = VkInstanceCreateInfo(
            pApplicationInfo=appInfo,
            ppEnabledLayerNames=enabledLayers,
            ppEnabledExtensionNames=extenstions
        )

        self._instance = vkCreateInstance(instanceInfo, None)
        InstanceProcAddr.T = self._instance

    def findDeviceAndQueues(self):
        physicalDevices = vkEnumeratePhysicalDevices(self._instance)
        if len(physicalDevices) == 0:
            raise Exception('No Physical device found.')

        self._physicalDevice = physicalDevices[0]

        self._queuesInfo.graphics.queueFamilyIndex = self.__getQueueFamilyIndex(VK_QUEUE_GRAPHICS_BIT)
        self._queuesInfo.compute.queueFamilyIndex = self.__getQueueFamilyIndex(VK_QUEUE_COMPUTE_BIT)
        self._queuesInfo.transfer.queueFamilyIndex = self.__getQueueFamilyIndex(VK_QUEUE_TRANSFER_BIT)

    def postCreateDevice(self):
        DeviceProcAddr.T = self._device
        self._queuesInfo.graphics.queue = vkGetDeviceQueue(self._device, self._queuesInfo.graphics.queueFamilyIndex, 0)
        self._queuesInfo.compute.queue = vkGetDeviceQueue(self._device, self._queuesInfo.compute.queueFamilyIndex, 0)
        self._queuesInfo.transfer.queue = vkGetDeviceQueue(self._device, self._queuesInfo.transfer.queueFamilyIndex, 0)

    def createDebugMessenger(self):
        if not self._settings.validationEnabled:
            return

        createInfo = VkDebugUtilsMessengerCreateInfoEXT(
            messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            pfnUserCallback=self.__debugCallback
        )

        self._debugMessenger = vkCreateDebugUtilsMessengerEXT(self._instance, createInfo, None)

    def createSurface(self):
        hwnd = self.winId()
        hinstance = Win32misc.getInstance(hwnd)
        createInfo = VkWin32SurfaceCreateInfoKHR(
            hinstance=hinstance,
            hwnd=hwnd
        )

        self._surface = vkCreateWin32SurfaceKHR(self._instance, createInfo, None)

        supportPresent = vkGetPhysicalDeviceSurfaceSupportKHR(self._physicalDevice, self._queuesInfo.graphics.queueFamilyIndex, self._surface)
        if not supportPresent:
            raise Exception('Graphics queue does not support presenting')

        surfaceFormats = vkGetPhysicalDeviceSurfaceFormatsKHR(self._physicalDevice, self._surface)

        if len(surfaceFormats) == 1 and surfaceFormats[0].format == VK_FORMAT_UNDEFINED:
            self._surfaceFormat = VkSurfaceFormatKHR(self._settings.desiredSurfaceFormat, surfaceFormats[0].colorSpace)
        else:
            found = False
            for sf in surfaceFormats:
                if sf.format == self._settings.desiredSurfaceFormat:
                    self._surfaceFormat = VkSurfaceFormatKHR(sf.format, sf.colorSpace)
                    found = True
                    break
            if not found:
                self._surfaceFormat = VkSurfaceFormatKHR(surfaceFormats[0].format, surfaceFormats[0].colorSpace)


    def createSwapchain(self):
        surfaceCapabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self._physicalDevice, self._surface)
        presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(self._physicalDevice, self._surface)

        presentMode = VK_PRESENT_MODE_FIFO_KHR
        prevSwapchain = self._swapchain

        createInfo = VkSwapchainCreateInfoKHR(
            surface=self._surface,
            minImageCount=surfaceCapabilities.minImageCount,
            imageFormat=self._surfaceFormat.format,
            imageColorSpace=self._surfaceFormat.colorSpace,
            imageExtent=VkExtent2D(self.width(), self.height()),
            imageArrayLayers=1,
            imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0,
            preTransform=VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=presentMode,
            clipped=VK_TRUE,
            oldSwapchain=prevSwapchain
        )

        self._swapchain = vkCreateSwapchainKHR(self._device, createInfo, None)

        if prevSwapchain:
            if len(self._swapchainImages) > 0:
                [vkDestroyImageView(self._device, iv, None) for iv in self._swapchainImages]
                self._swapchainImages = []
            vkDestroySwapchainKHR(self._device, prevSwapchain, None)

        self._swapchainImages = vkGetSwapchainImagesKHR(self._device, self._swapchain)

        for sim in self._swapchainImages:
            ivCreateInfo = VkImageViewCreateInfo(
                format=self._surfaceFormat.format,
                subresourceRange=VkImageSubresourceRange(
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
                ),
                viewType=VK_IMAGE_VIEW_TYPE_2D,
                image=sim,
                #components=[VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A]
            )
            iv = vkCreateImageView(self._device, ivCreateInfo, None)
            self._swapchainImageViews.append(iv)

    def createFences(self):
        if self._frameReadinessFences:
            self._frameReadinessFences = []

        createInfo = VkFenceCreateInfo(flags=VK_FENCE_CREATE_SIGNALED_BIT)

        for i in self._swapchainImageViews:
            fence = vkCreateFence(self._device, createInfo, None)
            self._frameReadinessFences.append(fence)

        self._bufferedFrameMaxNum = len(self._frameReadinessFences)

    def createOffsreenBuffers(self):
        usageFlags = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        self._offsreenImageResource.createImage(VK_IMAGE_TYPE_2D, self._surfaceFormat.format,
                                                VkExtent3D(self.width(), self.height(), 1), VK_IMAGE_TILING_OPTIMAL,
                                                usageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self._offsreenImageResource.createImageView(VK_IMAGE_VIEW_TYPE_2D, self._surfaceFormat.format,
                                                    [VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1])


    def createCommandPool(self):
        createInfo = VkCommandPoolCreateInfo(
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self._queuesInfo.graphics.queueFamilyIndex
        )
        self._commandPool = vkCreateCommandPool(self._device, createInfo, None)

    def createCommandBuffers(self):
        createInfo = VkCommandBufferAllocateInfo(
            commandPool=self._commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self._swapchainImages)
        )

        self._commandBuffers = vkAllocateCommandBuffers(self._device, createInfo)

    def createSynchronization(self):
        createInfo = VkSemaphoreCreateInfo(flags=0)

        self._imageAcquiredSemaphore = vkCreateSemaphore(self._device, createInfo, None)
        self._renderFinishedSemaphore = vkCreateSemaphore(self._device, createInfo, None)

    def imageBarrier(self, cmdBuffer, image, subresourceRange, srcAccessMask, dstAccessMask, oldLayout, newLayout):
        barrier = VkImageMemoryBarrier(
            srcAccessMask=srcAccessMask,
            dstAccessMask=dstAccessMask,
            oldLayout=oldLayout,
            newLayout=newLayout,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=subresourceRange
        )

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, None, 0, None, 1, barrier)

    def fillCommandBuffers(self):
        cmdBufferBeginInfo = VkCommandBufferBeginInfo(flags=0)
        subresourceRange = VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1
        )
        for i, cmdBuffer in enumerate(self._commandBuffers):
            vkBeginCommandBuffer(cmdBuffer, cmdBufferBeginInfo)

            self.imageBarrier(cmdBuffer, self._offsreenImageResource.image, subresourceRange,
                              0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL)
            self.recordCommandBufferForFrame(cmdBuffer, i)
            self.imageBarrier(cmdBuffer, self._swapchainImages[i], subresourceRange,
                              0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            self.imageBarrier(cmdBuffer, self._offsreenImageResource.image, subresourceRange,
                              VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)

            copyRegion = VkImageCopy(
                srcSubresource=[VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1],
                srcOffset=[0, 0, 0],
                dstSubresource=[VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1],
                dstOffset=[0, 0, 0],
                extent=[self.width(), self.height(), 1]
            )
            vkCmdCopyImage(cmdBuffer, self._offsreenImageResource.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           self._swapchainImages[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, copyRegion)
            self.imageBarrier(cmdBuffer, self._swapchainImages[i], subresourceRange,
                              VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
            vkEndCommandBuffer(cmdBuffer)

    def drawFrame(self):
        # if not self.isExposed():
        #     return
        UINT64_MAX = 18446744073709551615
        imageIndex = vkAcquireNextImageKHR(self._device, self._swapchain, UINT64_MAX,
                                           self._imageAcquiredSemaphore, None)
        fence = self._frameReadinessFences[imageIndex]
        vkWaitForFences(self._device, 1, [fence,], VK_TRUE, UINT64_MAX)
        vkResetFences(self._device, 1, [fence,])

        self.updateDataForFrame(imageIndex)

        submitInfo = VkSubmitInfo(
            # waitSemaphoreCount=1,
            pWaitSemaphores=[self._imageAcquiredSemaphore,],
            pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,],
            # commandBufferCount=1,
            pCommandBuffers=[self._commandBuffers[imageIndex],],
            # signalSemaphoreCount=1,
            pSignalSemaphores=[self._renderFinishedSemaphore,]
        )
        vkQueueSubmit(self._queuesInfo.graphics.queue, 1, [submitInfo,], fence)

        presentInfo = VkPresentInfoKHR(
            # waitSemaphoreCount=1,
            pWaitSemaphores=[self._renderFinishedSemaphore,],
            # swapchainCount=1,
            pSwapchains=[self._swapchain,],
            pImageIndices=[imageIndex,]
        )
        vkQueuePresentKHR(self._queuesInfo.graphics.queue, presentInfo)
        self.requestUpdate()

    def createDevice(self):
        queueFamilyIndexes = {}.fromkeys([self._queuesInfo.graphics.queueFamilyIndex,self._queuesInfo.compute.queueFamilyIndex,self._queuesInfo.transfer.queueFamilyIndex])
        queueCreateInfos = []
        for i in queueFamilyIndexes:
            queueCreateInfo = VkDeviceQueueCreateInfo(
                queueFamilyIndex=i,
                queueCount=1,
                pQueuePriorities=[0.0]
            )
            queueCreateInfos.append(queueCreateInfo)

        layerNames = []
        if self._settings.validationEnabled:
            layerNames.append('VK_LAYER_KHRONOS_validation')
        deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]
        features = VkPhysicalDeviceFeatures()

        createInfo = VkDeviceCreateInfo(
            pQueueCreateInfos=queueCreateInfos,
            # enabledLayerCount=0,
            ppEnabledLayerNames=layerNames,
            ppEnabledExtensionNames=deviceExtensions,
            pEnabledFeatures=features
        )

        self._device = vkCreateDevice(self._physicalDevice, createInfo, None)

    def init(self):
        pass

    def recordCommandBufferForFrame(self, comBuffer, frameIndex):
        pass

    def updateDataForFrame(self, frameIndex):
        pass

    def __getQueueFamilyIndex(self, queueFlag):
        familyProperties = vkGetPhysicalDeviceQueueFamilyProperties(self._physicalDevice)
        for i, prop in enumerate(familyProperties):
            if queueFlag == VK_QUEUE_COMPUTE_BIT:
                if prop.queueFlags & VK_QUEUE_COMPUTE_BIT and not (prop.queueFlags & VK_QUEUE_GRAPHICS_BIT):
                    return i
            elif queueFlag == VK_QUEUE_TRANSFER_BIT:
                if prop.queueFlags & VK_QUEUE_TRANSFER_BIT and \
                not (prop.queueFlags & VK_QUEUE_GRAPHICS_BIT) and \
                not (prop.queueFlags & VK_QUEUE_COMPUTE_BIT):
                    return i
            elif queueFlag == VK_QUEUE_GRAPHICS_BIT:
                if prop.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                    return i

        return -1

    def event(self, event):
        if event.type() == QtCore.QEvent.UpdateRequest:
            self.drawFrame()
        # elif event.type() == QtCore.QEvent.Expose:
        #     self.requestUpdate()

        return super(Application, self).event(event)


if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = Application()
    win.run()
    # win.show()


    # def clenaup():
    #     global win
    #     del win
    #
    # app.aboutToQuit.connect(clenaup)
    exit(app.exec_())
