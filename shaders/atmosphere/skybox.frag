// Copyright 2022-2026 Nikita Fediuchin. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Physically based atmosphere skybox.
// Based on this: https://github.com/sebh/UnrealEngineSkyAtmosphere

#define USE_CAMERA_VOLUME

spec const bool USE_CUBEMAP_ONLY = false;
spec const float SLICE_COUNT = 8.0f;
spec const float KM_PER_SLICE = 12.0f;

#include "atmosphere/common.gsl"
#include "common/depth.gsl"

pipelineState
{
	faceCulling = off;
}

in noperspective float2 fs.texCoords;
out float4 fb.color;

uniform sampler2D depthBuffer;

uniform sampler2D
{
	filter = linear;
} transLUT;
uniform sampler2D
{
	filter = linear;
} skyViewLUT;
uniform sampler3D
{
	filter = linear;
} cameraVolume;

uniform pushConstants
{
	float4x4 invViewProj;
	float3 cameraPos;
	float bottomRadius;
	float3 sunDir;
	float topRadius;
	float3 sunColor;
	float sunSize;
} pc;

//**********************************************************************************************************************
float2 skyViewToUv(bool intersectGround, float viewZenithCosAngle, float lightViewCosAngle, float viewHeight)
{
	float vHorizon = sqrt(viewHeight * viewHeight - pc.bottomRadius * pc.bottomRadius);
	float beta = acosFast4(vHorizon / viewHeight); float zenithHorizonAngle = M_PI - beta;
	float2 uv; uv.x = sqrt(fma(-lightViewCosAngle, 0.5f, 0.5f));

	if (!intersectGround)
	{
		float coord = acosFast4(viewZenithCosAngle) / zenithHorizonAngle;
		uv.y = (1.0f - sqrt(1.0f - coord)) * 0.5f;
	}
	else
	{
		float coord = (acosFast4(viewZenithCosAngle) - zenithHorizonAngle) / beta;
		uv.y = fma(sqrt(coord), 0.5f, 0.5f);
	}

	// Constrain uvs to valid sub texel range (avoid zenith derivative issue making LUT usage visible).
	const float2 skyViewSize = float2(textureSize(skyViewLUT, 0));
	return (uv + 0.5f / skyViewSize) * (skyViewSize / (skyViewSize + 1.0f));
}

float3 getSunLuminance(float3 worldDir, bool intersectGround)
{
	// Note: No early exit to smooth the sun disk.
	if (intersectGround)
		return float3(0.0f);
	float3 transmittance = getTransmittance(transLUT, Ray(pc.cameraPos, worldDir), pc.bottomRadius, pc.topRadius);
	float sunDisk = saturate(((dot(worldDir, pc.sunDir) - pc.sunSize) * 2.0f) / (1.0f - pc.sunSize));
	return transmittance * pc.sunColor * sunDisk;
}

void main()
{
	float depth = USE_CUBEMAP_ONLY ? FAR_PLANE_DEPTH : textureLod(depthBuffer, fs.texCoords, 0.0f).r;
	float3 worldDir = calcViewDirection(fs.texCoords, pc.invViewProj); float viewHeight = length(pc.cameraPos);

	if (viewHeight < pc.topRadius && depth == FAR_PLANE_DEPTH)
	{
		float3 upVector = normalize(pc.cameraPos); float viewZenithCosAngle = dot(worldDir, upVector);
		float3 sideVector = normalize(cross(upVector, worldDir));
		// Aligns toward the sun light but perpendicular to up vector.
		float3 forwardVector = normalize(cross(sideVector, upVector));
		float2 lightOnPlane = normalize(float2(dot(pc.sunDir, forwardVector), dot(pc.sunDir, sideVector)));
		float lightViewCosAngle = lightOnPlane.x;

		bool intersectGround = raycast(Sphere(float3(0.0f), pc.bottomRadius), Ray(pc.cameraPos, worldDir));
		float2 uv = skyViewToUv(intersectGround, viewZenithCosAngle, lightViewCosAngle, viewHeight);
		float3 skyColor = textureLod(skyViewLUT, uv, 0.0f).rgb + getSunLuminance(worldDir, intersectGround);
		fb.color = float4(min(skyColor, float3(FLOAT_BIG_16)), 1.0f);
		return;
	}

	if (USE_CUBEMAP_ONLY)
		return;

	float3 worldPos = calcWorldPosition(depth, fs.texCoords, pc.invViewProj);
	float slice = aerialPersDepthToSlice(length(worldPos * 0.001f));
	// We multiply by weight to fade to 0 at depth 0. That works for luminance and opacity.
	float weight = slice < 0.5f ? saturate(slice * 2.0f) : 1.0f; slice = max(slice, 0.5f);
	float w = sqrt(slice * (1.0f / SLICE_COUNT)); // Squared distribution
	fb.color = textureLod(cameraVolume, float3(fs.texCoords, w), 0.0f) * weight;
}