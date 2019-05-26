from application import *
from cffi import FFI


ffi2 = FFI()
ffi2.cdef('''
typedef struct 
{
    float transform[12];
    uint32_t instanceId : 24;
    uint32_t mask : 8;
    uint32_t instanceOffset : 24;
    uint32_t flags : 8;
    uint64_t accelerationStructureHandle;
} VkGeometryInstance;
''')


@InstanceProcAddr
def vkGetPhysicalDeviceFeatures2KHR(physicalDevice):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceProperties2KHR(physicalDevice, pNext=ffi.NULL):
    pass


# device extention
@DeviceProcAddr
def vkCreateAccelerationStructureNV(device, pCreateInfo, pAllocator):
    pass

@DeviceProcAddr
def vkDestroyAccelerationStructureNV(device, accelerationStructure, pAllocator):
    pass


@DeviceProcAddr
def vkGetAccelerationStructureMemoryRequirementsNV(device, pCreateInfo):
    pass


@DeviceProcAddr
def vkCmdCopyAccelerationStructureNV(commandBuffer, dst, src, mode):
    pass


@DeviceProcAddr
def vkBindAccelerationStructureMemoryNV(device, bindInfoCount, pBindInfos):
    pass


@DeviceProcAddr
def vkCmdBuildAccelerationStructureNV(commandBuffer, pInfo, instanceData, instanceOffset,
                                      update, dst, src, scratch, scratchOffset):
    pass


@DeviceProcAddr
def vkCmdTraceRaysNV(commandBuffer, raygenShaderBindingTableBuffer, raygenShaderBindingOffset,
                     missShaderBindingTableBuffer, missShaderBindingOffset, missShaderBindingStride,
                     hitShaderBindingTableBuffer, hitShaderBindingOffset, hitShaderBindingStride,
                     callableShaderBindingTableBuffer, callableShaderBindingOffset, callableShaderBindingStride,
                     width, height, depth):
    pass


@DeviceProcAddr
def vkGetRayTracingShaderGroupHandlesNV(device, pipeline, firstGroup, groupCount, dataSize, data=ffi.NULL):
    pass


@DeviceProcAddr
def vkCreateRayTracingPipelinesNV(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator):
    pass


@DeviceProcAddr
def vkGetAccelerationStructureHandleNV(device, accelerationStructure, dataSize):
    pass


class RaytracingApplication(Application):

    def __init__(self):
        super(RaytracingApplication, self).__init__()

        self._deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]
        self._rayTracingProperties = None

    def createDevice(self):
        queueFamilyIndexes = {}.fromkeys(
            [self._queuesInfo.graphics.queueFamilyIndex, self._queuesInfo.compute.queueFamilyIndex,
             self._queuesInfo.transfer.queueFamilyIndex])
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
        # self._deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]
        features2 = vkGetPhysicalDeviceFeatures2KHR(self._physicalDevice)

        createInfo = VkDeviceCreateInfo(
            pNext=features2,
            pQueueCreateInfos=queueCreateInfos,
            # enabledLayerCount=0,
            ppEnabledLayerNames=layerNames,
            ppEnabledExtensionNames=self._deviceExtensions,
        )

        self._device = vkCreateDevice(self._physicalDevice, createInfo, None)

    def initRayTracing(self):
        self._rayTracingProperties = VkPhysicalDeviceRayTracingPropertiesNV(
            maxRecursionDepth=0,
            shaderGroupHandleSize=0
        )

        props = vkGetPhysicalDeviceProperties2KHR(self._physicalDevice, self._rayTracingProperties)


if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = RaytracingApplication()
    win.run()

    exit(app.exec_())

