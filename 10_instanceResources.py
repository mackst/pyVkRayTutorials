import math
from raytracingApp import *


class TutorialApplication(RaytracingApplication):

    def __init__(self):
        super(TutorialApplication, self).__init__()

        self._appName = 'VkRay Tutorial 10: Instance resources (python)'

        self._topASMemory = None
        self._topAS = None
        self._bottomASMemory = None
        self._bottomAS = None
        self._rtDescriptorSetLayout = None
        self._rtPipelineLayout = None
        self._rtPipeline = None
        self._shaderBindingTable = BufferResource()
        self._rtDescriptorPool = None
        self._rtDescriptorSet = None
        self._hitShaderAndDataSize = 0

        self._instanceNum = 3
        self._uniformBuffers = []

        self._deviceExtensions.append(VK_NV_RAY_TRACING_EXTENSION_NAME)
        self._deviceExtensions.append(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)

    def __del__(self):
        if self._topAS:
            vkDestroyAccelerationStructureNV(self._device, self._topAS, None)
        if self._topASMemory:
            vkFreeMemory(self._device, self._topASMemory, None)
        if self._bottomAS:
            vkDestroyAccelerationStructureNV(self._device, self._bottomAS, None)
        if self._bottomASMemory:
            vkFreeMemory(self._device, self._bottomASMemory, None)

        if self._rtDescriptorPool:
            vkDestroyDescriptorPool(self._device, self._rtDescriptorPool, None)
        del self._shaderBindingTable

        if self._rtPipeline:
            vkDestroyPipeline(self._device, self._rtPipeline, None)
        if self._rtPipelineLayout:
            vkDestroyPipelineLayout(self._device, self._rtPipelineLayout, None)
        if self._rtDescriptorSetLayout:
            vkDestroyDescriptorSetLayout(self._device, self._rtDescriptorSetLayout, None)

        del self._uniformBuffers

        super(TutorialApplication, self).__del__()

    def init(self):
        self.initRayTracing()
        self.createAccelerationStructures()
        self.createPipeline()
        self.createShaderBindingTable()
        self.createUniformBuffers()
        self.createDescriptorSet()

    def recordCommandBufferForFrame(self, cmdBuffer, frameIndex):
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, self._rtPipeline)
        vkCmdBindDescriptorSets(
            cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, self._rtPipelineLayout,
            0, 1, [self._rtDescriptorSet, ], 0, None
        )

        vkCmdTraceRaysNV(
            cmdBuffer,
            self._shaderBindingTable.buffer, 0,
            self._shaderBindingTable.buffer, 1 * self._rayTracingProperties.shaderGroupHandleSize, self._rayTracingProperties.shaderGroupHandleSize,
            self._shaderBindingTable.buffer, 2 * self._rayTracingProperties.shaderGroupHandleSize, self._hitShaderAndDataSize,
            None, 0, 0,
            self.width(), self.height(), 1
        )

    def createAccelerationStructures(self):
        vertexBuffer = BufferResource()
        indexBuffer = BufferResource()

        scale = 0.25
        d = (1.0 + math.sqrt(5.0)) * 0.5 * scale

        vertices = np.array(
            [
                # Icosahedron vertices
                -scale, +d, 0,
                +scale, +d, 0,
                -scale, -d, 0,
                +scale, -d, 0,
                +0, -scale, +d,
                +0, +scale, +d,
                +0, -scale, -d,
                +0, +scale, -d,
                +d, 0, -scale,
                +d, 0, +scale,
                -d, 0, -scale,
                -d, 0, +scale,
            ],
            np.float32
        )

        vertexCount = int(len(vertices) / 3)
        vertexSize = 3 * vertices.itemsize
        vertexBufferSize = vertices.nbytes

        indices = np.array(
            [
                # Icosahedron indices
                0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
                1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
                4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
            ], np.uint16
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
            flags=VK_GEOMETRY_OPAQUE_BIT_NV,
            geometryType=VK_GEOMETRY_TYPE_TRIANGLES_NV,
            geometry=geometryData
        )
        geometries = [geometry, ]

        self._bottomAS, self._bottomASMemory = self.__createAccelerationStructure(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV, geometries, 0
        )

        instanceBuffer = BufferResource()

        accelerationStructureHandle = vkGetAccelerationStructureHandleNV(self._device, self._bottomAS, ffi.sizeof('uint64_t'))

        instances = ffi2.new('VkGeometryInstance []', self._instanceNum)
        for i in range(self._instanceNum):
            instances[i].instanceId = i
            instances[i].mask = 0xff
            instances[i].instanceOffset = i
            instances[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV
            instances[i].accelerationStructureHandle = accelerationStructureHandle
            instances[i].transform = [
                1.0, 0.0, 0.0, -1.5 + 1.5 * i,
                0.0, 1.0, 0.0, -0.5 + 0.5 * i,
                0.0, 0.0, 1.0, 0.0
            ]

        instanceBufferSize = ffi2.sizeof('VkGeometryInstance') * self._instanceNum
        instanceBuffer.create(instanceBufferSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        instanceBuffer.copyToBufferUsingMapUnmap(instances, instanceBufferSize)

        self._topAS, self._topASMemory = self.__createAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV, None, self._instanceNum)

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
            instanceCount=self._instanceNum,
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

    def createPipeline(self):
        accelerationStructureLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_RAYGEN_BIT_NV
        )

        outputImageLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_RAYGEN_BIT_NV
        )

        uniformBufferLayoutBinding = VkDescriptorSetLayoutBinding(
            binding=2,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=self._instanceNum,
            stageFlags=VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV
        )

        bindings = [accelerationStructureLayoutBinding, outputImageLayoutBinding, uniformBufferLayoutBinding]

        flags = [0, 0, VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT]
        bindingFlags = VkDescriptorSetLayoutBindingFlagsCreateInfoEXT(pBindingFlags=flags)

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            pNext=bindingFlags,
            pBindings=bindings
        )

        self._rtDescriptorSetLayout = vkCreateDescriptorSetLayout(self._device, layoutInfo, None)

        rgenShader = ShaderResource()
        chitShader = ShaderResource()
        missShader = ShaderResource()
        rgenShader.loadFromFile('rt_10_shaders.rgen.spv')
        chitShader.loadFromFile('rt_10_shaders.rchit.spv')
        missShader.loadFromFile('rt_10_shaders.rmiss.spv')

        shaderStages = [
            rgenShader.getShaderStage(VK_SHADER_STAGE_RAYGEN_BIT_NV),
            missShader.getShaderStage(VK_SHADER_STAGE_MISS_BIT_NV),
            chitShader.getShaderStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV),
        ]

        pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
            pSetLayouts=[self._rtDescriptorSetLayout, ],
            pushConstantRangeCount=0
        )

        self._rtPipelineLayout = vkCreatePipelineLayout(self._device, pipelineLayoutCreateInfo, None)

        VK_SHADER_UNUSED_NV = 4294967295
        shaderGroups = [
            # group0 = [ raygen ]
            VkRayTracingShaderGroupCreateInfoNV(
                type=VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV,
                generalShader=0,
                closestHitShader=VK_SHADER_UNUSED_NV,
                anyHitShader=VK_SHADER_UNUSED_NV,
                intersectionShader=VK_SHADER_UNUSED_NV
            ),
            # group1 = [ miss ]
            VkRayTracingShaderGroupCreateInfoNV(
                type=VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV,
                generalShader=1,
                closestHitShader=VK_SHADER_UNUSED_NV,
                anyHitShader=VK_SHADER_UNUSED_NV,
                intersectionShader=VK_SHADER_UNUSED_NV
            ),
            # group2 = [ chit ]
            VkRayTracingShaderGroupCreateInfoNV(
                type=VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV,
                generalShader=VK_SHADER_UNUSED_NV,
                closestHitShader=2,
                anyHitShader=VK_SHADER_UNUSED_NV,
                intersectionShader=VK_SHADER_UNUSED_NV
            ),
        ]

        rayPipelineInfo = VkRayTracingPipelineCreateInfoNV(
            pStages=shaderStages,
            pGroups=shaderGroups,
            layout=self._rtPipelineLayout,
            maxRecursionDepth=1,
            basePipelineIndex=0
        )

        self._rtPipeline = vkCreateRayTracingPipelinesNV(self._device, None, 1, [rayPipelineInfo, ], None)

    def createShaderBindingTable(self):
        inlineDataSize = ffi.sizeof('float') * 4
        self._hitShaderAndDataSize = self._rayTracingProperties.shaderGroupHandleSize + inlineDataSize
        raygenAndMissSize = self._rayTracingProperties.shaderGroupHandleSize * 2
        shaderBindingTableSize = raygenAndMissSize + self._hitShaderAndDataSize * self._instanceNum

        self._shaderBindingTable.create(shaderBindingTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)

        mappedMemory = self._shaderBindingTable.map(shaderBindingTableSize)
        mm_array = np.frombuffer(mappedMemory, np.uint8)
        buf = ffi.from_buffer(mappedMemory, require_writable=True)
        bufdata = ffi.cast('uint8_t*', buf)
        rtsgHandles = vkGetRayTracingShaderGroupHandlesNV(self._device, self._rtPipeline, 0, 2, raygenAndMissSize, buf)
        bufdataprt = bufdata + raygenAndMissSize
        mindex = raygenAndMissSize

        # colors = np.array(
        #     [
        #         [0.5, 0, 0, 0],
        #         [0.0, 0.5, 0, 0],
        #         [0.0, 0, 0.5, 0],
        #     ], np.float32
        # )
        colors = [
            [127, 0, 0, 0],
            [0, 127, 0, 0],
            [0, 0, 127, 0],
        ]

        for i in range(self._instanceNum):
            handles = vkGetRayTracingShaderGroupHandlesNV(self._device, self._rtPipeline, 2, 1,
                                                          self._rayTracingProperties.shaderGroupHandleSize, bufdataprt)

            bufdataprt = bufdataprt + self._rayTracingProperties.shaderGroupHandleSize
            mindex += self._rayTracingProperties.shaderGroupHandleSize

            for x in range(4):
                mm_array[mindex+x] = colors[i][0]
                mm_array[mindex+4+x] = colors[i][1]
                mm_array[mindex+8+x] = colors[i][2]
                mm_array[mindex+12+x] = colors[i][3]
            # bufdataprt[0] = colors[i][0]
            # bufdataprt[1] = colors[i][1]
            # bufdataprt[2] = colors[i][2]
            # bufdataprt[3] = colors[i][3]
            bufdataprt += inlineDataSize
            mindex += inlineDataSize

        self._shaderBindingTable.unmap()
    
    def createDescriptorSet(self):
        poolSizes = [
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, self._instanceNum),
        ]

        descriptorPoolCreateInfo = VkDescriptorPoolCreateInfo(
            maxSets=1,
            pPoolSizes=poolSizes
        )

        self._rtDescriptorPool = vkCreateDescriptorPool(self._device, descriptorPoolCreateInfo, None)

        variableDescriptorCountInfo = VkDescriptorSetVariableDescriptorCountAllocateInfoEXT(
            descriptorSetCount=1,
            pDescriptorCounts=[self._instanceNum, ]
        )

        descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
            pNext=variableDescriptorCountInfo,
            descriptorPool=self._rtDescriptorPool,
            pSetLayouts=[self._rtDescriptorSetLayout, ]
        )

        self._rtDescriptorSet = vkAllocateDescriptorSets(self._device, descriptorSetAllocateInfo)[0]

        descriptorAccelerationStructureInfo = VkWriteDescriptorSetAccelerationStructureNV(
            pAccelerationStructures=[self._topAS, ]
        )

        accelerationStructureWrite = VkWriteDescriptorSet(
            pNext=descriptorAccelerationStructureInfo,
            dstSet=self._rtDescriptorSet,
            dstBinding=0,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV
        )

        descriptorOutputImageInfo = VkDescriptorImageInfo(
            imageView=self._offsreenImageResource.imageView,
            imageLayout=VK_IMAGE_LAYOUT_GENERAL
        )

        outputImageWrite = VkWriteDescriptorSet(
            dstSet=self._rtDescriptorSet,
            dstBinding=1,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=descriptorOutputImageInfo
        )

        bufferInfos = [
            VkDescriptorBufferInfo(
                buffer=i.buffer,
                offset=0,
                range=i.size
            ) for i in self._uniformBuffers
        ]

        uniformBuffers = VkWriteDescriptorSet(
            dstSet=self._rtDescriptorSet,
            dstBinding=2,
            dstArrayElement=0,
            descriptorCount=len(bufferInfos),
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=bufferInfos
        )
        descriptorWrites = [accelerationStructureWrite, outputImageWrite, uniformBuffers]
        vkUpdateDescriptorSets(self._device, len(descriptorWrites), descriptorWrites, 0, None)

    def createUniformBuffers(self):
        colors = np.array(
            [
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ], np.float32
        )

        memoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT

        for i in range(self._instanceNum):
            buf = BufferResource()
            buf.create(ffi.sizeof('float') * 3, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, memoryFlags)
            buf.copyToBufferUsingMapUnmap(ffi.cast('float*', colors[i].ctypes.data), colors[i].nbytes)
            self._uniformBuffers.append(buf)

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

