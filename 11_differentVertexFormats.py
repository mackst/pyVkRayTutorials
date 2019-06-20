import math

from raytracingApp import *


VK_WHOLE_SIZE = 18446744073709551615


class RenderObject(object):

    def __init__(self):
        self.geometry = None
        self.bottomAS = None
        self.bottomASMemory = None
        self.vertexBuffers = (BufferResource(), BufferResource(), BufferResource())
        self.indexBuffer = BufferResource()
        self.indexBufferCopy = BufferResource()
        self.uniformBuffer = BufferResource()
        self.texture = ImageResource()
        self.vertexNum = 0
        self.indexNum = 0
        self.shaderIndex = 0


class UniformBufferContent(object):

    def __init__(self):
        self.vertexBufferArrayOffset = 0
        self.indexBufferArrayOffset = 0
        self.textureArrayOffset = 0
        self.padding = 0

    def tolist(self):
        return [
            self.vertexBufferArrayOffset, self.indexBufferArrayOffset, self.textureArrayOffset, self.padding
        ]

    # @property
    # def data(self):
    #     a = np.array([
    #         self.vertexBufferArrayOffset, self.indexBufferArrayOffset, self.textureArrayOffset, self.padding
    #     ], np.uint32)
    #     return a.ctypes.data

    @property
    def nbytes(self):
        return ffi.sizeof('uint32_t') * 4


class TutorialApplication(RaytracingApplication):

    def __init__(self):
        super(TutorialApplication, self).__init__()

        self._appName = 'VkRay Tutorial 11: Different Vertex Formats (python)'

        self._topASMemory = None
        self._topAS = None
        self._rtPipelineLayout = None
        self._rtPipeline = None
        self._shaderBindingTable = BufferResource()
        self._rtDescriptorPool = None

        self._rtDescriptorSetLayouts = []
        self._rtDescriptorSets = []
        self._objectNum = 5
        self._renderObjects = (RenderObject(), RenderObject(), RenderObject(), RenderObject(), RenderObject())
        self._vertexBufferViews = []
        self._indexBufferViews = []
        self._imageViews = []
        self._samplers = []

        self._deviceExtensions.append(VK_NV_RAY_TRACING_EXTENSION_NAME)
        self._deviceExtensions.append(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)

    def __del__(self):
        if self._topAS:
            vkDestroyAccelerationStructureNV(self._device, self._topAS, None)
        if self._topASMemory:
            vkFreeMemory(self._device, self._topASMemory, None)
        for i in self._vertexBufferViews:
            vkDestroyBufferView(self._device, i, None)
        for i in self._indexBufferViews:
            vkDestroyBufferView(self._device, i, None)
        for obj in self._renderObjects:
            if obj.bottomAS:
                vkDestroyAccelerationStructureNV(self._device, obj.bottomAS, None)
            if obj.bottomASMemory:
                vkFreeMemory(self._device, obj.bottomASMemory, None)
            del obj.vertexBuffers
            del obj.indexBuffer
            del obj.uniformBuffer
            del obj.texture

        if self._rtDescriptorPool:
            vkDestroyDescriptorPool(self._device, self._rtDescriptorPool, None)
        del self._shaderBindingTable

        if self._rtPipeline:
            vkDestroyPipeline(self._device, self._rtPipeline, None)
        if self._rtPipelineLayout:
            vkDestroyPipelineLayout(self._device, self._rtPipelineLayout, None)
        for i in self._rtDescriptorSetLayouts:
            vkDestroyDescriptorSetLayout(self._device, i, None)

        super(TutorialApplication, self).__del__()

    def init(self):
        self.initRayTracing()

        self.createBox(self._renderObjects[0], 'cb0.bmp')
        self.createIcosahedron(self._renderObjects[1])
        self.createBox(self._renderObjects[2], 'cb1.bmp')
        self.createBox(self._renderObjects[3], 'cb2.bmp')
        self.createIcosahedron(self._renderObjects[4])

        self.createAccelerationStructures()
        self.createDescriptorSetLayouts()
        self.createPipeline()
        self.createShaderBindingTable()
        self.createPoolAndAllocateDescriptorSets()
        self.updateDescriptorSets()

    def recordCommandBufferForFrame(self, cmdBuffer, frameIndex):
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, self._rtPipeline)
        vkCmdBindDescriptorSets(
            cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, self._rtPipelineLayout,
            0, len(self._rtDescriptorSets), self._rtDescriptorSets, 0, None
        )

        vkCmdTraceRaysNV(
            cmdBuffer,
            self._shaderBindingTable.buffer, 0,
            self._shaderBindingTable.buffer, 1 * self._rayTracingProperties.shaderGroupHandleSize, self._rayTracingProperties.shaderGroupHandleSize,
            self._shaderBindingTable.buffer, 2 * self._rayTracingProperties.shaderGroupHandleSize, self._rayTracingProperties.shaderGroupHandleSize,
            None, 0, 0,
            self.width(), self.height(), 1
        )

    def _createBufferAndUploadData(self, buffer, usage, content):
        memoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT

        buffer.create(content.nbytes, usage, memoryFlags)
        if content.dtype == np.float32:
            dt = 'float*'
        elif content.dtype == np.uint16:
            dt = 'uint16_t*'
        elif content.dtype == np.uint32:
            dt = 'uint32_t*'
        data = ffi.cast(dt, content.ctypes.data)
        buffer.copyToBufferUsingMapUnmap(data, content.nbytes)

    def createAccelerationStructures(self):
        instanceBuffer = BufferResource()

        instances = ffi2.new('VkGeometryInstance []', self._objectNum)
        width = 4.0
        height = 0.75
        depth = 0.75
        stepX = width / self._objectNum
        stepY = height / self._objectNum
        stepZ = depth / self._objectNum
        biasX = -stepX * (self._objectNum - 1) * 0.5
        biasY = -stepY * (self._objectNum - 1) * 0.5 - 0.75
        biasZ = -stepZ * (self._objectNum - 1) * 0.5

        for i, instance in enumerate(instances):
            accelerationStructureHandle = vkGetAccelerationStructureHandleNV(
                self._device, self._renderObjects[i].bottomAS, ffi.sizeof('uint64_t')
            )
            instance.instanceId = i
            instance.mask = 0xff
            instance.instanceOffset = self._renderObjects[i].shaderIndex
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV
            instance.accelerationStructureHandle = accelerationStructureHandle
            instance.transform = [
                1.0, 0.0, 0.0, biasX + stepX * i,
                0.0, 1.0, 0.0, biasY + stepY * i,
                0.0, 0.0, 1.0, biasZ + stepZ * i,
            ]

        instanceBufferSize = ffi2.sizeof('VkGeometryInstance') * self._objectNum
        instanceBuffer.create(instanceBufferSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        instanceBuffer.copyToBufferUsingMapUnmap(instances, instanceBufferSize)

        self._topAS, self._topASMemory = self.__createAccelerationStructure(
            VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV, None, self._objectNum
        )

        bottomAccelerationStructureBufferSize = max([self.__getScratchBufferSize(obj.bottomAS) for obj in self._renderObjects])
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

        for obj in self._renderObjects:
            asInfo = VkAccelerationStructureInfoNV(
                type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV,
                instanceCount=0,
                pGeometries=obj.geometry
            )

            vkCmdBuildAccelerationStructureNV(
                commandBuffer, asInfo, None, 0, VK_FALSE, obj.bottomAS, None, scratchBuffer.buffer, 0
            )

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                                 0, 1, [memoryBarrier, ], 0, None, 0, None)

        asInfo = VkAccelerationStructureInfoNV(
            type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV,
            instanceCount=self._objectNum,
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

    def createDescriptorSetLayouts(self):
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
            descriptorCount=self._objectNum,
            stageFlags=VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV
        )

        bindings = [accelerationStructureLayoutBinding, outputImageLayoutBinding, uniformBufferLayoutBinding]

        flags = [0, 0, VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT]

        bindingFlags = VkDescriptorSetLayoutBindingFlagsCreateInfoEXT(pBindingFlags=flags)

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            pNext=bindingFlags,
            pBindings=bindings
        )

        dsl = vkCreateDescriptorSetLayout(self._device, layoutInfo, None)
        self._rtDescriptorSetLayouts.append(dsl)

        flag = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT
        bindingFlags = VkDescriptorSetLayoutBindingFlagsCreateInfoEXT(
            pBindingFlags=[flag, ]
        )

        texelBufferBinding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
            descriptorCount=self._objectNum * 2,
            stageFlags=VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV
        )

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            pNext=bindingFlags,
            pBindings=[texelBufferBinding, ]
        )

        dsl = vkCreateDescriptorSetLayout(self._device, layoutInfo, None)
        self._rtDescriptorSetLayouts.append(dsl)

        texelBufferBinding.descriptorCount = self._objectNum

        dsl = vkCreateDescriptorSetLayout(self._device, layoutInfo, None)
        self._rtDescriptorSetLayouts.append(dsl)

        flag = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT
        bindingFlags = VkDescriptorSetLayoutBindingFlagsCreateInfoEXT(
            pBindingFlags=[flag, ]
        )

        texelBufferBinding = VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=self._objectNum,
            stageFlags=VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV
        )

        layoutInfo = VkDescriptorSetLayoutCreateInfo(
            pNext=bindingFlags,
            pBindings=[texelBufferBinding, ]
        )

        dsl = vkCreateDescriptorSetLayout(self._device, layoutInfo, None)
        self._rtDescriptorSetLayouts.append(dsl)

    def createPipeline(self):
        rgenShader = ShaderResource()
        chitShaders = (ShaderResource(), ShaderResource())
        missShader = ShaderResource()
        rgenShader.loadFromFile('rt_11_shaders.rgen.spv')
        missShader.loadFromFile('rt_11_shaders.rmiss.spv')
        chitShaders[0].loadFromFile('rt_11_box.rchit.spv')
        chitShaders[1].loadFromFile('rt_11_icosahedron.rchit.spv')

        shaderStages = [
            rgenShader.getShaderStage(VK_SHADER_STAGE_RAYGEN_BIT_NV), 
            missShader.getShaderStage(VK_SHADER_STAGE_MISS_BIT_NV),
            chitShaders[0].getShaderStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV),
            chitShaders[1].getShaderStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV)
        ]

        pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
            pSetLayouts=self._rtDescriptorSetLayouts,
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
            # group3 = [ chit ]
            VkRayTracingShaderGroupCreateInfoNV(
                type=VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV,
                generalShader=VK_SHADER_UNUSED_NV,
                closestHitShader=3,
                anyHitShader=VK_SHADER_UNUSED_NV,
                intersectionShader=VK_SHADER_UNUSED_NV
            ),
        ]

        rayPipelineInfo = VkRayTracingPipelineCreateInfoNV(
            pStages=shaderStages,
            pGroups=shaderGroups,
            maxRecursionDepth=1,
            layout=self._rtPipelineLayout,
            basePipelineIndex=0
        )

        self._rtPipeline = vkCreateRayTracingPipelinesNV(self._device, None, 1, [rayPipelineInfo, ], None)

    def createShaderBindingTable(self):
        groupNum = 4
        shaderBindingTableSize = self._rayTracingProperties.shaderGroupHandleSize * groupNum

        self._shaderBindingTable.create(shaderBindingTableSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)

        mappedMemory = self._shaderBindingTable.map(shaderBindingTableSize)
        buf = ffi.from_buffer(mappedMemory, require_writable=True)
        rtsgHandles = vkGetRayTracingShaderGroupHandlesNV(self._device, self._rtPipeline, 0, groupNum, shaderBindingTableSize, buf)
        self._shaderBindingTable.unmap()
    
    def createPoolAndAllocateDescriptorSets(self):
        poolSizes = [
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, self._objectNum),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, self._objectNum * 3),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, self._objectNum),
        ]

        descriptorPoolCreateInfo = VkDescriptorPoolCreateInfo(
            maxSets=len(self._rtDescriptorSetLayouts),
            pPoolSizes=poolSizes
        )

        self._rtDescriptorPool = vkCreateDescriptorPool(self._device, descriptorPoolCreateInfo, None)

        variableDescriptorCounts = [
            self._objectNum,
            len(self._vertexBufferViews),
            len(self._indexBufferViews),
            len(self._imageViews)
        ]

        variableDescriptorCountInfo = VkDescriptorSetVariableDescriptorCountAllocateInfoEXT(
            descriptorSetCount=len(self._rtDescriptorSetLayouts),
            pDescriptorCounts=variableDescriptorCounts
        )

        descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
            pNext=variableDescriptorCountInfo,
            descriptorPool=self._rtDescriptorPool,
            pSetLayouts=self._rtDescriptorSetLayouts
        )

        self._rtDescriptorSets = vkAllocateDescriptorSets(self._device, descriptorSetAllocateInfo)

    def updateDescriptorSets(self):
        descriptorAccelerationStructureInfo = VkWriteDescriptorSetAccelerationStructureNV(
            pAccelerationStructures=[self._topAS, ]
        )

        accelerationStructureWrite = VkWriteDescriptorSet(
            pNext=descriptorAccelerationStructureInfo,
            dstSet=self._rtDescriptorSets[0],
            dstBinding=0,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV,
        )

        descriptorOutputImageInfo = VkDescriptorImageInfo(
            imageView=self._offsreenImageResource.imageView,
            imageLayout=VK_IMAGE_LAYOUT_GENERAL
        )

        outputImageWrite = VkWriteDescriptorSet(
            dstSet=self._rtDescriptorSets[0],
            dstBinding=1,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=descriptorOutputImageInfo
        )

        bufferInfo = [
            VkDescriptorBufferInfo(
                buffer=obj.uniformBuffer.buffer,
                offset=0,
                range=obj.uniformBuffer.size
            ) for obj in self._renderObjects
        ]

        uniformBuffers = VkWriteDescriptorSet(
            dstSet=self._rtDescriptorSets[0],
            dstBinding=2,
            dstArrayElement=0,
            descriptorCount=len(bufferInfo),
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=bufferInfo
        )

        vertexBuffers = VkWriteDescriptorSet(
            dstSet=self._rtDescriptorSets[1],
            dstBinding=0,
            dstArrayElement=0,
            descriptorCount=len(self._vertexBufferViews),
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
            pTexelBufferView=self._vertexBufferViews
        )

        indexBuffers = VkWriteDescriptorSet(
            dstSet=self._rtDescriptorSets[2],
            dstBinding=0,
            dstArrayElement=0,
            descriptorCount=len(self._indexBufferViews),
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
            pTexelBufferView=self._indexBufferViews
        )

        imageInfoArray = [
            VkDescriptorImageInfo(
                sampler=self._samplers[i],
                imageView=self._imageViews[i],
                imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            ) for i in range(len(self._imageViews))
        ]

        imageWrite = VkWriteDescriptorSet(
            dstSet=self._rtDescriptorSets[3],
            dstBinding=0,
            dstArrayElement=0,
            descriptorCount=len(imageInfoArray),
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            pImageInfo=imageInfoArray
        )

        descriptorWrites = [
            accelerationStructureWrite, outputImageWrite, uniformBuffers,
            vertexBuffers, indexBuffers, imageWrite
        ]

        vkUpdateDescriptorSets(self._device, len(descriptorWrites), descriptorWrites, 0, None)

    def createIcosahedron(self, obj):
        obj.shaderIndex = 1

        content = UniformBufferContent()
        content.vertexBufferArrayOffset = len(self._vertexBufferViews)
        content.indexBufferArrayOffset = len(self._indexBufferViews)
        content.textureArrayOffset = len(self._imageViews)

        memoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        obj.uniformBuffer.create(content.nbytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, memoryFlags)
        a = np.array(content.tolist(), np.uint32)
        obj.uniformBuffer.copyToBufferUsingMapUnmap(a, content.nbytes)

        self.createIcosahedronGeometry(obj)
        self.createIcosahedronBufferViews(obj)
        self.__createObjectBottomLevelAS(obj)

    def createBox(self, obj, texturePath):
        obj.shaderIndex = 0

        content = UniformBufferContent()
        content.vertexBufferArrayOffset = len(self._vertexBufferViews)
        content.indexBufferArrayOffset = len(self._indexBufferViews)
        content.textureArrayOffset = len(self._imageViews)

        memoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        obj.uniformBuffer.create(content.nbytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, memoryFlags)
        a = np.array(content.tolist(), np.uint32)
        obj.uniformBuffer.copyToBufferUsingMapUnmap(a, content.nbytes)

        self.createBoxGeometry(obj)
        self.createBoxBufferViews(obj)
        self.loadObjectTexture(obj, texturePath)
        self.__createObjectBottomLevelAS(obj)

    def createIcosahedronGeometry(self, obj):
        scale = 0.25
        d = (1.0 + math.sqrt(5.0)) * .5 * scale

        positions = np.array([
            [-scale, +d, 0],
            [+scale, +d, 0],
            [-scale, -d, 0],
            [+scale, -d, 0],
            [+0, -scale, +d],
            [+0, +scale, +d],
            [+0, -scale, -d],
            [+0, +scale, -d],
            [+d, 0, -scale],
            [+d, 0, +scale],
            [-d, 0, -scale],
            [-d, 0, +scale]
        ], np.float32)

        self._createBufferAndUploadData(obj.vertexBuffers[0], VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, positions)

        normals = np.zeros_like(positions)
        for i, norm in enumerate(normals):
            pos = positions[i]
            invLength = 1.0 / math.sqrt(np.sum(pos * pos))
            normals[i] = pos * invLength

        self._createBufferAndUploadData(obj.vertexBuffers[1], VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT, normals)

        indices = np.array([
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
        ], np.uint16)

        self._createBufferAndUploadData(obj.indexBuffer, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices)

        indicesWithPadding = np.array([[indices[i], indices[i+1], indices[i+2], 0] for i in range(0, len(indices), 3)],
                                      np.uint16)

        self._createBufferAndUploadData(obj.indexBufferCopy, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT, indicesWithPadding)

        obj.vertexNum = len(positions)
        obj.indexNum = len(indices)

    def createBoxGeometry(self, obj):
        boxHalfSize = 0.25

        positions = np.array([
            [-boxHalfSize, -boxHalfSize, -boxHalfSize], [-boxHalfSize, -boxHalfSize, boxHalfSize],
            [-boxHalfSize, boxHalfSize, -boxHalfSize], [-boxHalfSize, boxHalfSize, boxHalfSize],
            [boxHalfSize, -boxHalfSize, -boxHalfSize], [boxHalfSize, -boxHalfSize, boxHalfSize],
            [boxHalfSize, boxHalfSize, -boxHalfSize], [boxHalfSize, boxHalfSize, boxHalfSize],
            [-boxHalfSize, -boxHalfSize, -boxHalfSize], [-boxHalfSize, -boxHalfSize, boxHalfSize],
            [boxHalfSize, -boxHalfSize, -boxHalfSize], [boxHalfSize, -boxHalfSize, boxHalfSize],
            [-boxHalfSize, boxHalfSize, -boxHalfSize], [-boxHalfSize, boxHalfSize, boxHalfSize],
            [boxHalfSize, boxHalfSize, -boxHalfSize], [boxHalfSize, boxHalfSize, boxHalfSize],
            [-boxHalfSize, -boxHalfSize, -boxHalfSize], [-boxHalfSize, boxHalfSize, -boxHalfSize],
            [boxHalfSize, -boxHalfSize, -boxHalfSize], [boxHalfSize, boxHalfSize, -boxHalfSize],
            [-boxHalfSize, -boxHalfSize, boxHalfSize], [-boxHalfSize, boxHalfSize, boxHalfSize],
            [boxHalfSize, -boxHalfSize, boxHalfSize], [boxHalfSize, boxHalfSize, boxHalfSize],
        ], np.float32)

        self._createBufferAndUploadData(obj.vertexBuffers[0], VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, positions)

        texcoords = np.array([
            0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
            0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
            0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
            0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
            0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
            0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
        ], np.float32)

        self._createBufferAndUploadData(obj.vertexBuffers[1], VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT, texcoords)

        normals = np.array([
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
        ], np.float32)

        self._createBufferAndUploadData(obj.vertexBuffers[2], VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT, normals)

        indices = np.array([
            0, 1, 2, 1, 2, 3,
            4, 5, 6, 5, 6, 7,
            8, 9, 10, 9, 10, 11,
            12, 13, 14, 13, 14, 15,
            16, 17, 18, 17, 18, 19,
            20, 21, 22, 21, 22, 23
        ], np.uint16)

        self._createBufferAndUploadData(obj.indexBuffer, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices)

        indicesWithPadding = np.array(
            [[indices[i], indices[i + 1], indices[i + 2], 0] for i in range(0, len(indices), 3)],
            np.uint16)

        self._createBufferAndUploadData(obj.indexBufferCopy, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
                                        indicesWithPadding)

        obj.vertexNum = len(positions)
        obj.indexNum = len(indices)

    def createIcosahedronBufferViews(self, obj):
        bufferViewInfo = VkBufferViewCreateInfo(
            buffer=obj.vertexBuffers[1].buffer,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=0,
            range=VK_WHOLE_SIZE
        )

        vertexBufferView = vkCreateBufferView(self._device, bufferViewInfo, None)

        bufferViewInfo.buffer = obj.indexBufferCopy.buffer
        bufferViewInfo.format = VK_FORMAT_R16G16B16A16_UINT

        indexBufferView = vkCreateBufferView(self._device, bufferViewInfo, None)

        self._vertexBufferViews.append(vertexBufferView)
        self._indexBufferViews.append(indexBufferView)

    def createBoxBufferViews(self, obj):
        bufferViewInfo = VkBufferViewCreateInfo(
            buffer=obj.vertexBuffers[1].buffer,
            format=VK_FORMAT_R32G32_SFLOAT,
            offset=0,
            range=VK_WHOLE_SIZE
        )

        vertexBufferView1 = vkCreateBufferView(self._device, bufferViewInfo, None)

        bufferViewInfo.buffer = obj.vertexBuffers[2].buffer
        bufferViewInfo.format = VK_FORMAT_R32G32B32_SFLOAT

        vertexBufferView2 = vkCreateBufferView(self._device, bufferViewInfo, None)

        bufferViewInfo.buffer = obj.indexBufferCopy.buffer
        bufferViewInfo.format = VK_FORMAT_R16G16B16A16_UINT

        indexBufferView = vkCreateBufferView(self._device, bufferViewInfo, None)

        self._vertexBufferViews.append(vertexBufferView1)
        self._vertexBufferViews.append(vertexBufferView2)
        self._indexBufferViews.append(indexBufferView)

    def loadObjectTexture(self, obj, imPath):
        obj.texture.loadTexture2DFromFile(imPath)

        subresourceRange = VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=VK_REMAINING_MIP_LEVELS,
            baseArrayLayer=0,
            layerCount=VK_REMAINING_ARRAY_LAYERS
        )

        obj.texture.createImageView(VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB, subresourceRange)
        obj.texture.createSampler(VK_FILTER_NEAREST, VK_FILTER_LINEAR,
                                  VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT)

        self._imageViews.append(obj.texture.imageView)
        self._samplers.append(obj.texture.sampler)

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

    def __createObjectBottomLevelAS(self, obj):
        triangles = VkGeometryTrianglesNV(
            vertexData=obj.vertexBuffers[0].buffer,
            vertexOffset=0,
            vertexCount=obj.vertexNum,
            vertexStride=12,
            vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
            indexData=obj.indexBuffer.buffer,
            indexOffset=0,
            indexCount=obj.indexNum,
            indexType=VK_INDEX_TYPE_UINT16,
            transformOffset=0
        )

        geometryData = VkGeometryDataNV(triangles, VkGeometryAABBNV())

        geometry = VkGeometryNV(
            geometryType=VK_GEOMETRY_TYPE_TRIANGLES_NV,
            geometry=geometryData
        )
        obj.geometry = [geometry, ]
        obj.bottomAS, obj.bottomASMemory = self.__createAccelerationStructure(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV, obj.geometry, 0
        )

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

