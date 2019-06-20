from application import *



@InstanceProcAddr
def vkGetPhysicalDeviceFeatures2KHR(physicalDevice):
    pass


@InstanceProcAddr
def vkGetPhysicalDeviceProperties2KHR(physicalDevice, pProperties=None):
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
def vkGetRayTracingShaderGroupHandlesNV(device, pipeline, firstGroup, groupCount, dataSize):
    pass


@DeviceProcAddr
def vkCreateRayTracingPipelinesNV(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator):
    pass


@DeviceProcAddr
def vkGetAccelerationStructureHandleNV(device, accelerationStructure, dataSize):
    pass


class TutorialApplication(Application):

    def __init__(self):
        super(TutorialApplication, self).__init__()

        self._appName = 'VkRay Tutorial 01: Initialization (python)'
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
        deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_NV_RAY_TRACING_EXTENSION_NAME]
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
        vkCreateAccelerationStructureNV = vkGetDeviceProcAddr(self._device, 'vkCreateAccelerationStructureNV')
        vkDestroyAccelerationStructureNV = vkGetDeviceProcAddr(self._device, 'vkDestroyAccelerationStructureNV')
        vkGetAccelerationStructureMemoryRequirementsNV = vkGetDeviceProcAddr(self._device, 'vkGetAccelerationStructureMemoryRequirementsNV')
        vkCmdCopyAccelerationStructureNV = vkGetDeviceProcAddr(self._device, 'vkCmdCopyAccelerationStructureNV')
        vkBindAccelerationStructureMemoryNV = vkGetDeviceProcAddr(self._device, 'vkBindAccelerationStructureMemoryNV')
        vkCmdBuildAccelerationStructureNV = vkGetDeviceProcAddr(self._device, 'vkCmdBuildAccelerationStructureNV')
        vkCmdTraceRaysNV = vkGetDeviceProcAddr(self._device, 'vkCmdTraceRaysNV')
        vkGetRayTracingShaderGroupHandlesNV = vkGetDeviceProcAddr(self._device, 'vkGetRayTracingShaderGroupHandlesNV')
        vkCreateRayTracingPipelinesNV = vkGetDeviceProcAddr(self._device, 'vkCreateRayTracingPipelinesNV')
        vkGetAccelerationStructureHandleNV = vkGetDeviceProcAddr(self._device, 'vkGetAccelerationStructureHandleNV')


        self._rayTracingProperties = VkPhysicalDeviceRayTracingPropertiesNV(
            maxRecursionDepth=0,
            shaderGroupHandleSize=0
        )

        props = VkPhysicalDeviceProperties2(pNext=self._rayTracingProperties)
        props = vkGetPhysicalDeviceProperties2KHR(self._physicalDevice, props)


if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = TutorialApplication()
    win.run()
    # win.show()

    exit(app.exec_())

