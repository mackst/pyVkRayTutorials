import math
from raytracingApp import *


class TutorialApplication(RaytracingApplication):

    def __init__(self):
        super(TutorialApplication, self).__init__()

        self._appName = 'VkRay Tutorial 07: Instance buffer - more instances and geometries (python)'

        self._topASMemory = None
        self._topAS = None
        self._bottomASMemory = []
        self._bottomAS = []
        self._rtDescriptorSetLayout = None
        self._rtPipelineLayout = None
        self._rtPipeline = None
        self._shaderBindingTable = BufferResource()
        self._rtDescriptorPool = None
        self._rtDescriptorSet = None

        self._deviceExtensions.append(VK_NV_RAY_TRACING_EXTENSION_NAME)

    def __del__(self):
        if self._topAS:
            vkDestroyAccelerationStructureNV(self._device, self._topAS, None)
        if self._topASMemory:
            vkFreeMemory(self._device, self._topASMemory, None)
        [vkDestroyAccelerationStructureNV(self._device, i, None) for i in self._bottomAS if i]
        [vkFreeMemory(self._device, i, None) for i in self._bottomASMemory if i]

        if self._rtDescriptorPool:
            vkDestroyDescriptorPool(self._device, self._rtDescriptorPool, None)
        del self._shaderBindingTable

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
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, self._rtPipeline)
        vkCmdBindDescriptorSets(
            cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, self._rtPipelineLayout,
            0, 1, [self._rtDescriptorSet, ], 0, None
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
        vertexBuffer = BufferResource()
        indexBuffer = BufferResource()

        scale = 0.25
        d = (1.0 + math.sqrt(5.0)) * 0.5 * scale

        vertices = np.array(
            [
                # Triangle vertices
                -0.5, -0.5, 0.0,
                0.0, 0.5, 0.0,
                0.5, -0.5, 0.0

                # Tutorial 07 vertices
                -10.0, 0.0, 10.0,
                10.0, 0.0, 10.0,
                10.0, 0.0, -10.0,
                -10.0, 0.0, -10.0,

                # Icosahedron vertices
                -scale, d, 0.0,
                scale, d, 0,
                -scale, -d, 0,
                scale, -d, 0,

                0, -scale, d,
                0, scale, d,
                0, -scale, -d,
                0, scale, -d,

                d, 0, -scale,
                d, 0, scale,
                -d, 0, -scale,
                -d, 0, scale,
            ],
            np.float32
        )

        vertexCount =  int(len(vertices) / 3)
        vertexSize = 3 * vertices.itemsize
        vertexBufferSize = vertices.nbytes

        indices = np.array(
            [
                # Triangle indices
                0, 1, 2,
                # Tutorial 07 indices
                # Quad indices
                0, 1, 2, 2, 3, 0,
                # Icosahedron indices
                0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
                1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
                4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
            ], 
            np.uint16
        )

        indexCount = len(indices)
        indexSize = indices.itemsize
        indexBufferSize = indices.nbytes

        vertexBuffer.create(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        vertexBuffer.copyToBufferUsingMapUnmap(ffi.cast('float*', vertices.ctypes.data), vertexBufferSize)

        indexBuffer.create(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        indexBuffer.copyToBufferUsingMapUnmap(ffi.cast('uint16_t*', indices.ctypes.data), indexBufferSize)

        geometries = [
            # Insert single triangle
            VkGeometryNV(
                geometryType=VK_GEOMETRY_TYPE_TRIANGLES_NV,
                geometry=VkGeometryDataNV(
                    VkGeometryTrianglesNV(
                        vertexData=vertexBuffer.buffer,
                        vertexOffset=0,
                        vertexCount=3,
                        vertexStride=vertexSize,
                        vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
                        indexData=indexBuffer.buffer,
                        indexOffset=0,
                        indexCount=3,
                        indexType=VK_INDEX_TYPE_UINT16,
                        transformOffset=0
                    ),
                    VkGeometryAABBNV()
                )
            ),
            # Insert bottom quad, use data from same vertex/index buffers, but with offset
            VkGeometryNV(
                geometryType=VK_GEOMETRY_TYPE_TRIANGLES_NV,
                geometry=VkGeometryDataNV(
                    VkGeometryTrianglesNV(
                        vertexData=vertexBuffer.buffer,
                        vertexOffset=3 * vertexSize,
                        vertexCount=4,
                        vertexStride=vertexSize,
                        vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
                        indexData=indexBuffer.buffer,
                        indexOffset=3 * indexSize,
                        indexCount=6,
                        indexType=VK_INDEX_TYPE_UINT16,
                        transformOffset=0
                    ),
                    VkGeometryAABBNV()
                )
            ),
            # Insert icosahedron, use data from same vertex/index buffers, but with offset
            VkGeometryNV(
                geometryType=VK_GEOMETRY_TYPE_TRIANGLES_NV,
                geometry=VkGeometryDataNV(
                    VkGeometryTrianglesNV(
                        vertexData=vertexBuffer.buffer,
                        vertexOffset=7 * vertexSize,
                        vertexCount=12,
                        vertexStride=vertexSize,
                        vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
                        indexData=indexBuffer.buffer,
                        indexOffset=9 * indexSize,
                        indexCount=60,
                        indexType=VK_INDEX_TYPE_UINT16,
                        transformOffset=0
                    ),
                    VkGeometryAABBNV()
                )
            ),
        ]

        for geometry in geometries:
            bottomAS, bottomASMemory = self.__createAccelerationStructure(
                VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV, [geometry,], 0
            )
            self._bottomAS.append(bottomAS)
            self._bottomASMemory.append(bottomASMemory)

        instanceBuffer = BufferResource()

        accelerationStructureHandle = [vkGetAccelerationStructureHandleNV(self._device, i, ffi.sizeof('uint64_t')) for i in self._bottomAS]

        transform = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        instances = ffi2.new('VkGeometryInstance[5]')
        # 3 instances of the bottom level AS #1
        instances[0].transform = transform
        instances[0].instanceId = 0
        instances[0].mask = 0xff
        instances[0].instanceOffset = 0
        instances[0].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV
        instances[0].accelerationStructureHandle = accelerationStructureHandle[0]

        transform[3] = 1.5
        transform[11] = 0.5
        instances[1].transform = transform
        instances[1].instanceId = 1
        instances[1].mask = 0xff
        instances[1].instanceOffset = 0
        instances[1].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV
        instances[1].accelerationStructureHandle = accelerationStructureHandle[0]

        transform[3] = -1.5
        transform[11] = -0.5
        instances[2].transform = transform
        instances[2].instanceId = 2
        instances[2].mask = 0xff
        instances[2].instanceOffset = 0
        instances[2].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV
        instances[2].accelerationStructureHandle = accelerationStructureHandle[0]

        # 1 instance of the bottom level AS #2
        transform = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, -9.0,
            0.0, 0.0, 2.0, 0.0
        ]
        instances[3].transform = transform
        instances[3].instanceId = 3
        instances[3].mask = 0xff
        instances[3].instanceOffset = 0
        instances[3].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV
        instances[3].accelerationStructureHandle = accelerationStructureHandle[1]

        # 1 instance of the bottom level AS #3
        transform[3] = 3.5
        transform[11] = 0.5
        instances[4].transform = transform
        instances[4].instanceId = 4
        instances[4].mask = 0xff
        instances[4].instanceOffset = 0
        instances[4].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV
        instances[4].accelerationStructureHandle = accelerationStructureHandle[2]
        # instance = ffi2.new('VkGeometryInstance *', [
        #     [
        #         1.0, 0.0, 0.0, 0.0,
        #         0.0, 1.0, 0.0, 0.0,
        #         0.0, 0.0, 1.0, 0.0
        #     ], 0, 0xff, 0, VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV, accelerationStructureHandle
        # ])

        instanceNum = len(instances)
        instanceBufferSize = ffi2.sizeof(instances)
        instanceBuffer.create(instanceBufferSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        instanceBuffer.copyToBufferUsingMapUnmap(instances, instanceBufferSize)

        self._topAS, self._topASMemory = self.__createAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV, None, instanceNum)

        bottomAccelerationStructureBufferSize = max([self.__getScratchBufferSize(i) for i in self._bottomAS])
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

        for i in range(len(self._bottomAS)):
            asInfo = VkAccelerationStructureInfoNV(
                type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV,
                instanceCount=0,
                pGeometries=[geometries[i],]
            )

            vkCmdBuildAccelerationStructureNV(
                commandBuffer, asInfo, None, 0, VK_FALSE, self._bottomAS[i], None, scratchBuffer.buffer, 0
            )

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                                0, 1, [memoryBarrier, ], 0, None, 0, None)

        asInfo = VkAccelerationStructureInfoNV(
            type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV,
            instanceCount=instanceNum,
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

        bindings = [accelerationStructureLayoutBinding, outputImageLayoutBinding]

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            pBindings=bindings
        )

        self._rtDescriptorSetLayout = vkCreateDescriptorSetLayout(self._device, layoutInfo, None)

        rgenShader = ShaderResource()
        chitShader = ShaderResource()
        missShader = ShaderResource()
        ahitShader = ShaderResource()
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
        poolSizes = [
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1),
        ]

        descriptorPoolCreateInfo = VkDescriptorPoolCreateInfo(
            maxSets=1,
            pPoolSizes=poolSizes
        )

        self._rtDescriptorPool = vkCreateDescriptorPool(self._device, descriptorPoolCreateInfo, None)

        descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
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
        descriptorWrites = [accelerationStructureWrite, outputImageWrite]
        vkUpdateDescriptorSets(self._device, len(descriptorWrites), descriptorWrites, 0, None)

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

