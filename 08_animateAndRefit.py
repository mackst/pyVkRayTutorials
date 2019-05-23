import math
from raytracingApp import *


class Frame(object):

    def __init__(self):
        self.topASMemory = None
        self.topAS = None
        self.bottomASMemory = None
        self.bottomAS = None
        self.rtDescriptorSet = None
        self.instanceBuffer = BufferResource()
        self.vertexBuffer = BufferResource()
        self.indexBuffer = BufferResource()
        self.bottomASHandle = -1
        self.geometries = []


class TutorialApplication(RaytracingApplication):

    def __init__(self):
        super(TutorialApplication, self).__init__()

        self._appName = 'VkRay Tutorial 08: Animate and refit (python)'

        self._frames = []

        self._rtPipelineLayout = None
        self._rtPipeline = None
        self._shaderBindingTable = BufferResource()
        self._scratchBuffer = BufferResource()
        self._rtDescriptorPool = None
        self._rtDescriptorSet = None

        self.vertexNum = 3
        self.indexNum = 3

        self._deviceExtensions.append(VK_NV_RAY_TRACING_EXTENSION_NAME)

    def __del__(self):
        for frame in self._frames:
            if frame.topAS:
                vkDestroyAccelerationStructureNV(self._device, frame.topAS, None)
            if frame.topASMemory:
                vkFreeMemory(self._device, frame.topASMemory, None)
            if frame.bottomAS:
                vkDestroyAccelerationStructureNV(self._device, frame.bottomAS, None)
            if frame.bottomASMemory:
                vkFreeMemory(self._device, frame.bottomASMemory, None)

            del frame.instanceBuffer
            del frame.indexBuffer
            del frame.vertexBuffer

        if self._rtDescriptorPool:
            vkDestroyDescriptorPool(self._device, self._rtDescriptorPool, None)
        del self._shaderBindingTable
        del self._scratchBuffer

        if self._rtPipeline:
            vkDestroyPipeline(self._device, self._rtPipeline, None)
        if self._rtPipelineLayout:
            vkDestroyPipelineLayout(self._device, self._rtPipelineLayout, None)
        if self._rtDescriptorSetLayout:
            vkDestroyDescriptorSetLayout(self._device, self._rtDescriptorSetLayout, None)

        super(TutorialApplication, self).__del__()

    def init(self):
        self.initRayTracing()
        self.createAccelerationStructures()
        self.createPipeline()
        self.createShaderBindingTable()
        self.createDescriptorSet()

    def recordCommandBufferForFrame(self, cmdBuffer, frameIndex):
        frame = self._frames[frameIndex]
        memoryBarrier = VkMemoryBarrier(
            srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
            dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV
        )

        asInfo = VkAccelerationStructureInfoNV(
            type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV,
            instanceCount=0,
            pGeometries=frame.geometries
        )

        vkCmdBuildAccelerationStructureNV(
            cmdBuffer, asInfo, None, 0, VK_TRUE, frame.bottomAS, frame.bottomAS, self._scratchBuffer.buffer, 0
        )

        vkCmdPipelineBarrier(
            cmdBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0, 1, [memoryBarrier, ], 0, None, 0, None
        )

        asInfo = VkAccelerationStructureInfoNV(
            type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV,
            instanceCount=1,
            geometryCount=0
        )

        vkCmdBuildAccelerationStructureNV(
            cmdBuffer, asInfo, frame.instanceBuffer.buffer, 0, VK_TRUE, frame.topAS, frame.topAS, self._scratchBuffer.buffer, 0
        )

        vkCmdPipelineBarrier(
            cmdBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV, 0, 1, [memoryBarrier, ], 0, None, 0, None
        )

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, self._rtPipeline)
        vkCmdBindDescriptorSets(
            cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, self._rtPipelineLayout,
            0, 1, [frame.rtDescriptorSet, ], 0, None
        )

        vkCmdTraceRaysNV(
            cmdBuffer,
            self._shaderBindingTable.buffer, 0,
            self._shaderBindingTable.buffer, 2 * self._rayTracingProperties.shaderGroupHandleSize, self._rayTracingProperties.shaderGroupHandleSize,
            self._shaderBindingTable.buffer, 1 * self._rayTracingProperties.shaderGroupHandleSize, self._rayTracingProperties.shaderGroupHandleSize,
            None, 0, 0,
            self.width(), self.height(), 1
        )

    def createAccelerationStructures(self):
        for i in range(self._bufferedFrameMaxNum):
            frame = Frame()

            indexSize = ffi.sizeof('uint16_t')
            indexBufferSize = self.indexNum * indexSize

            vertexSize = 3 * ffi.sizeof('float')
            vertexBufferSize = self.vertexNum * vertexSize

            frame.vertexBuffer.create(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            frame.indexBuffer.create(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

            frame.geometries = [
                VkGeometryNV(
                    geometryType=VK_GEOMETRY_TYPE_TRIANGLES_NV,
                    geometry=VkGeometryDataNV(
                        VkGeometryTrianglesNV(
                            vertexData=frame.vertexBuffer.buffer,
                            vertexOffset=0,
                            vertexCount=self.vertexNum,
                            vertexStride=vertexSize,
                            vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
                            indexData=frame.indexBuffer.buffer,
                            indexOffset=0,
                            indexCount=self.indexNum,
                            indexType=VK_INDEX_TYPE_UINT16,
                            transformOffset=0
                        ),
                        VkGeometryAABBNV()
                    )
                ),
            ]

            self.__fillVertexBuffer(frame, 0.0)

            indices = np.array(
                [0, 1, 2], np.uint16
            )

            frame.indexBuffer.copyToBufferUsingMapUnmap(ffi.cast('uint16_t*', indices.ctypes.data), indices.nbytes)

            self._frames.append(frame)

        # for frame in self._frames:
            frame.bottomAS, frame.bottomASMemory = self.__createAccelerationStructure(
                VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV, frame.geometries, 0
            )

            frame.bottomASHandle = vkGetAccelerationStructureHandleNV(self._device, frame.bottomAS, ffi.sizeof('uint64_t'))

            instanceBufferSize = ffi2.sizeof('VkGeometryInstance')
            frame.instanceBuffer.create(instanceBufferSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            self.__fillInstanceBuffer(frame, 0.0)

            frame.topAS, frame.topASMemory = self.__createAccelerationStructure(
                VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV, None, 1
            )

        scratchBufferSize = 0

        for frame in self._frames:
            bottomAccelerationStructureBufferSize = self.__getScratchBufferSize(frame.bottomAS)
            topAccelerationStructureBufferSize = self.__getScratchBufferSize(frame.topAS)
            maxSize = max(topAccelerationStructureBufferSize, bottomAccelerationStructureBufferSize)
            scratchBufferSize = max(scratchBufferSize, maxSize)

        self._scratchBuffer.create(scratchBufferSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

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

        for frame in self._frames:
            asInfo = VkAccelerationStructureInfoNV(
                type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV,
                flags=VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV,
                instanceCount=0,
                pGeometries=frame.geometries
            )

            vkCmdBuildAccelerationStructureNV(
                commandBuffer, asInfo, None, 0, VK_FALSE, frame.bottomAS, None, self._scratchBuffer.buffer, 0
            )

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                                0, 1, [memoryBarrier, ], 0, None, 0, None)

            asInfo = VkAccelerationStructureInfoNV(
                type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV,
                flags=VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV,
                instanceCount=1,
                geometryCount=0
            )

            vkCmdBuildAccelerationStructureNV(
                commandBuffer, asInfo, frame.instanceBuffer.buffer, 0, VK_FALSE, frame.topAS, None,
                self._scratchBuffer.buffer, 0
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

        bindings = [accelerationStructureLayoutBinding, outputImageLayoutBinding]

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            pBindings=bindings
        )

        self._rtDescriptorSetLayout = vkCreateDescriptorSetLayout(self._device, layoutInfo, None)

        rgenShader = ShaderResource()
        chitShader = ShaderResource()
        missShader = ShaderResource()
        # ahitShader = ShaderResource()
        rgenShader.loadFromFile('rt_06_shaders.rgen.spv')
        chitShader.loadFromFile('rt_06_shaders.rchit.spv')
        missShader.loadFromFile('rt_06_shaders.rmiss.spv')

        shaderStages = [
            rgenShader.getShaderStage(VK_SHADER_STAGE_RAYGEN_BIT_NV), 
            chitShader.getShaderStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV),
            missShader.getShaderStage(VK_SHADER_STAGE_MISS_BIT_NV)
        ]

        pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
            pSetLayouts=[self._rtDescriptorSetLayout, ],
            pushConstantRangeCount=0
        )

        self._rtPipelineLayout = vkCreatePipelineLayout(self._device, pipelineLayoutCreateInfo, None)

        shaderGroups = [
            # group0 = [ raygen ]
            VkRayTracingShaderGroupCreateInfoNV(
                type=VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV,
                generalShader=0,
                closestHitShader=VK_SHADER_UNUSED_NV,
                anyHitShader=VK_SHADER_UNUSED_NV,
                intersectionShader=VK_SHADER_UNUSED_NV
            ),
            # group1 = [ chit ]
            VkRayTracingShaderGroupCreateInfoNV(
                type=VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV,
                generalShader=VK_SHADER_UNUSED_NV,
                closestHitShader=1,
                anyHitShader=VK_SHADER_UNUSED_NV,
                intersectionShader=VK_SHADER_UNUSED_NV
            ),
            # group2 = [ miss ]
            VkRayTracingShaderGroupCreateInfoNV(
                type=VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV,
                generalShader=2,
                closestHitShader=VK_SHADER_UNUSED_NV,
                anyHitShader=VK_SHADER_UNUSED_NV,
                intersectionShader=VK_SHADER_UNUSED_NV
            )
        ]

        rayPipelineInfo = VkRayTracingPipelineCreateInfoNV(
            pStages=shaderStages,
            pGroups=shaderGroups,
            layout=self._rtPipelineLayout,
            basePipelineIndex=0
        )

        self._rtPipeline = vkCreateRayTracingPipelinesNV(self._device, None, 1, [rayPipelineInfo, ], None)

    def createShaderBindingTable(self):
        groupNum = 3
        shaderBindingTableSize = self._rayTracingProperties.shaderGroupHandleSize * groupNum

        self._shaderBindingTable.create(shaderBindingTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)

        # mappedMemory = self._shaderBindingTable.map(shaderBindingTableSize)
        mappedMemory = vkGetRayTracingShaderGroupHandlesNV(self._device, self._rtPipeline, 0, groupNum, shaderBindingTableSize)
        # self._shaderBindingTable.unmap()
    
    def createDescriptorSet(self):
        frameNum = len(self._frames)
        poolSizes = [
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, frameNum),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, frameNum),
        ]

        descriptorPoolCreateInfo = VkDescriptorPoolCreateInfo(
            maxSets=frameNum,
            pPoolSizes=poolSizes
        )

        self._rtDescriptorPool = vkCreateDescriptorPool(self._device, descriptorPoolCreateInfo, None)

        descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
            descriptorPool=self._rtDescriptorPool,
            pSetLayouts=[self._rtDescriptorSetLayout, ]
        )

        for frame in self._frames:
            frame.rtDescriptorSet = vkAllocateDescriptorSets(self._device, descriptorSetAllocateInfo)[0]

            descriptorAccelerationStructureInfo = VkWriteDescriptorSetAccelerationStructureNV(
                pAccelerationStructures=[frame.topAS, ]
            )

            accelerationStructureWrite = VkWriteDescriptorSet(
                pNext=descriptorAccelerationStructureInfo,
                dstSet=frame.rtDescriptorSet,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV
            )

            descriptorOutputImageInfo = VkDescriptorImageInfo(
                imageView=self._offsreenImageResource.imageView,
                imageLayout=VK_IMAGE_LAYOUT_GENERAL
            )

            outputImageWrite = VkWriteDescriptorSet(
                dstSet=frame.rtDescriptorSet,
                dstBinding=1,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=descriptorOutputImageInfo
            )
            descriptorWrites = [accelerationStructureWrite, outputImageWrite]
            vkUpdateDescriptorSets(self._device, len(descriptorWrites), descriptorWrites, 0, None)

    def __createAccelerationStructure(self, asType, geometries, instanceCount):
        info = VkAccelerationStructureInfoNV(
            type=asType,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV,
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

    def __updateDataFrame(self, frameIndex):
        time = 0

        self.__fillVertexBuffer(self._frames[frameIndex], time)
        self.__fillInstanceBuffer(self._frames[frameIndex], time)

    def __fillVertexBuffer(self, frame, time):
        scale = math.sin(time * 5.0) * 0.5 + 1.0
        bias = math.sin(time * 3.0) * 0.5

        vertices = np.array(
            [
                -0.5 * scale + bias, -0.5 * scale, 0.0,
                0.0 + bias, 0.5 * scale, 0.0,
                0.5 * scale + bias, -0.5 * scale, 0.0
            ],
            np.float32
        )

        frame.vertexBuffer.copyToBufferUsingMapUnmap(ffi.cast('float*', vertices.ctypes.data), vertices.nbytes)

    def __fillInstanceBuffer(self, frame, time):
        instance = ffi2.new('VkGeometryInstance *', [
            [
                math.cos(time), -math.sin(time), 0.0, 0.0,
                math.sin(time), math.cos(time), 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0
            ], 0, 0xff, 0, VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV, frame.bottomASHandle
        ])

        frame.indexBuffer.copyToBufferUsingMapUnmap(instance, ffi.sizeof(instance))




if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)

    win = TutorialApplication()
    win.run()

    exit(app.exec_())

