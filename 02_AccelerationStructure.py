from raytracingApp import *


class TutorialApplication(RaytracingApplication):

    def __init__(self):
        super(TutorialApplication, self).__init__()

        self._appName = 'VkRay Tutorial 02: Building Acceleration Structure (python)'

        self._topASMemory = None
        self._topAS = None
        self._bottomASMemory = None
        self._bottomAS = None

        self._deviceExtensions.append(VK_NV_RAY_TRACING_EXTENSION_NAME)

    def __del__(self):
        if self._topAS:
            vkDestroyAccelerationStructureNV(self._device, self._topAS, None)
        if self._topASMemory:
            vkFreeMemory(self._device, self._topASMemory, None)
        if self._bottomAS:
            vkDestroyAccelerationStructureNV(self._device, self._bottomAS, None)
        if self._bottomASMemory:
            vkFreeMemory(self._device, self._bottomASMemory, None)

        super(TutorialApplication, self).__del__()

    def init(self):
        self.initRayTracing()
        self.createAccelerationStructures()

    def createAccelerationStructures(self):
        vertexBuffer = BufferResource()
        indexBuffer = BufferResource()

        vertices = np.array(
            [
                -0.5, -0.5, 0.0,
                0.0, 0.5, 0.0,
                0.5, -0.5, 0.0
            ],
            np.float32
        )

        vertexCount = 3
        vertexSize = 3 * vertices.itemsize
        vertexBufferSize = vertices.nbytes

        indices = np.array(
            [0, 1, 2], np.uint16
        )

        indexCount = len(indices)
        indexSize = indices.itemsize
        indexBufferSize = indices.nbytes

        vertexBuffer.create(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        vertexBuffer.copyToBufferUsingMapUnmap(ffi.cast('float*', vertices.ctypes.data), vertexBufferSize)

        indexBuffer.create(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        indexBuffer.copyToBufferUsingMapUnmap(ffi.cast('uint16_t*', indices.ctypes.data), indexBufferSize)

        triangles = VkGeometryTrianglesNV(
            vertexData=vertexBuffer.buffer,
            vertexOffset=0,
            vertexCount=vertexCount,
            vertexStride=vertexSize,
            vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
            indexData=indexBuffer.buffer,
            indexOffset=0,
            indexCount=indexCount,
            indexType=VK_INDEX_TYPE_UINT16,
            transformOffset=0
        )

        geometryData = VkGeometryDataNV(triangles, VkGeometryAABBNV())

        geometry = VkGeometryNV(
            geometryType=VK_GEOMETRY_TYPE_TRIANGLES_NV,
            geometry=geometryData
        )
        geometries = [geometry, ]

        self._bottomAS, self._bottomASMemory = self.__createAccelerationStructure(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV, geometries, 0
        )

        instanceBuffer = BufferResource()

        accelerationStructureHandle = vkGetAccelerationStructureHandleNV(self._device, self._bottomAS, ffi.sizeof('uint64_t'))

        instance = ffi2.new('VkGeometryInstance *', [
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0
            ], 0, 0xff, 0, VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV, accelerationStructureHandle
        ])

        instanceBufferSize = ffi2.sizeof('VkGeometryInstance')
        instanceBuffer.create(instanceBufferSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        instanceBuffer.copyToBufferUsingMapUnmap(instance, instanceBufferSize)

        self._topAS, self._topASMemory = self.__createAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV, None, 1)

        bottomAccelerationStructureBufferSize = self.__getScratchBufferSize(self._bottomAS)
        topAccelerationStructureBufferSize = self.__getScratchBufferSize(self._topAS)
        scratchBufferSize = max(bottomAccelerationStructureBufferSize, topAccelerationStructureBufferSize)

        scratchBuffer = BufferResource()
        scratchBuffer.create(scratchBufferSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

        commandBufferAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self._commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        commandBuffer = vkAllocateCommandBuffers(self._device, commandBufferAllocateInfo)[0]

        beginInfo = VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vkBeginCommandBuffer(commandBuffer, beginInfo)

        memoryBarrier = VkMemoryBarrier(
            srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
            dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV
        )

        asInfo = VkAccelerationStructureInfoNV(
            type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV,
            instanceCount=0,
            pGeometries=geometries
        )

        vkCmdBuildAccelerationStructureNV(
            commandBuffer, asInfo, None, 0, VK_FALSE, self._bottomAS, None, scratchBuffer.buffer, 0
        )

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                             0, 1, [memoryBarrier, ], 0, None, 0, None)

        asInfo = VkAccelerationStructureInfoNV(
            type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV,
            instanceCount=1,
            geometryCount=0
        )

        vkCmdBuildAccelerationStructureNV(
            commandBuffer, asInfo, instanceBuffer.buffer, 0, VK_FALSE, self._topAS, None, scratchBuffer.buffer, 0
        )

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV,
                             0, 1, [memoryBarrier, ], 0, None, 0, None)

        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(
            waitSemaphoreCount=0,
            pCommandBuffers=[commandBuffer, ],
            signalSemaphoreCount=0
        )

        vkQueueSubmit(self._queuesInfo.graphics.queue, 1, [submitInfo, ], None)
        vkQueueWaitIdle(self._queuesInfo.graphics.queue)
        vkFreeCommandBuffers(self._device, self._commandPool, 1, [commandBuffer, ])

    def __createAccelerationStructure(self, asType, geometries, instanceCount):
        info = VkAccelerationStructureInfoNV(
            type=asType,
            instanceCount=instanceCount,
            geometryCount=0
        )
        if geometries:
            info = VkAccelerationStructureInfoNV(
                type=asType,
                instanceCount=instanceCount,
                pGeometries=geometries
            )
        accelerationStructureInfo = VkAccelerationStructureCreateInfoNV(
            compactedSize=0,
            info=info
        )

        AS = vkCreateAccelerationStructureNV(self._device, accelerationStructureInfo, None)

        memoryRequirementsInfo = VkAccelerationStructureMemoryRequirementsInfoNV(
            accelerationStructure=AS,
            type=VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV
        )

        memoryRequirements = vkGetAccelerationStructureMemoryRequirementsNV(self._device, memoryRequirementsInfo)

        memoryAllocateInfo = VkMemoryAllocateInfo(
            allocationSize=memoryRequirements.memoryRequirements.size,
            memoryTypeIndex=ResourceBase.getMemoryType(memoryRequirements.memoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        )
        ASMemory = vkAllocateMemory(self._device, memoryAllocateInfo, None)

        bindInfo = VkBindAccelerationStructureMemoryInfoNV(
            accelerationStructure=AS,
            memory=ASMemory,
            memoryOffset=0,
            deviceIndexCount=0
        )

        vkBindAccelerationStructureMemoryNV(self._device, 1, [bindInfo, ])

        return (AS, ASMemory)

    def __getScratchBufferSize(self, handle):
        memoryRequirementsInfo = VkAccelerationStructureMemoryRequirementsInfoNV(
            accelerationStructure=handle,
            type=VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV
        )
        memoryRequirements = vkGetAccelerationStructureMemoryRequirementsNV(self._device, memoryRequirementsInfo)

        return memoryRequirements.memoryRequirements.size




if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = TutorialApplication()
    win.run()

    exit(app.exec_())

