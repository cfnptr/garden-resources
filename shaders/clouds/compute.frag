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

// Physically based volumetric clouds.
// Based on the Nubis clouds rendering system.

#define USE_CAMERA_VOLUME

spec const float STEP_ADJ_DIST = 16.384f;
spec const float SLICE_COUNT = 8.0f;
spec const float KM_PER_SLICE = 12.0f;

#include "clouds/common.gsl"
#include "common/depth.gsl"
#include "common/constants.gsl"
#include "atmosphere/common.gsl"

pipelineState
{
	faceCulling = off;
}

in noperspective float2 fs.texCoords;

out float4 fb.color;
out float4 fb.depth;

uniform sampler2D hizBuffer;

uniform sampler3D
{
	filter = linear;
} cameraVolume;
uniform sampler2D
{
	addressMode = repeat;
	filter = linear;
} dataFields;
uniform sampler2D
{
	filter = linear;
} verticalProfile;
uniform sampler3D
{
	addressMode = repeat;
	filter = linear;
} noiseShape;

uniform CommonConstants
{
	COMMON_CONSTANTS
} cc;

uniform pushConstants
{
	float3 cameraPos;
	float bottomRadius;
	float topRadius;
	float minDistance;
	float maxDistance;
	float currentTime;
	float coverage;
	float temperature;
} pc;

float calcStepSize(float distance)
{
	const float nearStepSize = 0.003f, farStepOffset = 0.06f;
	return fma(farStepOffset * distance, STEP_ADJ_DIST, nearStepSize);
}

//**********************************************************************************************************************
void main()
{
	Ray ray = Ray(pc.cameraPos, calcViewDirection(fs.texCoords, cc.invViewProj));
	float2 tRay = raycast2(Sphere(float3(0.0f), pc.topRadius), ray);

	if (!isIntersected(tRay)) // No atmosphere intersection so no clouds.
	{
		fb.color = float4(0.0f); fb.depth = float4(0.0f);
		return;
	}
	tRay.x = max(tRay.x, pc.minDistance); 

	float2 tBottom = raycast2(Sphere(float3(0.0f), pc.bottomRadius), ray);
	if (isIntersected(tBottom)) // Intesecting bottom clouds sphere.
	{
		tRay = tBottom.x < 0.0f ? // Is camera below bottom clouds level?
			float2(max(tRay.x, tBottom.y), tRay.y) : float2(tRay.x, min(tRay.y, tBottom.x)); 
	}

	float depth = textureLod(hizBuffer, fs.texCoords, 0.0f).r;
	float3 worldPos = calcWorldPosition(depth, fs.texCoords, cc.invViewProj);
	tRay.y = depth > 0.0f ? min(tRay.y, length(worldPos * 0.001f)) : tRay.y;

	// Skipping ray if whole planet is behind us.
	if (tRay.y <= tRay.x || tRay.x > pc.maxDistance)
	{
		fb.color = float4(0.0f); fb.depth = float4(0.0f);
		return;
	}

	float sunIntens = dot(cc.sunLight, cc.sunLight);
	float invThickness = 1.0f / (pc.topRadius - pc.bottomRadius);
	float3 fieldWindDir = calcFieldWindDir(cc.windDir, pc.currentTime);
	float3 noiseWindDir = calcNoiseWindDir(cc.windDir, pc.currentTime);
	float3 sunDir = -cc.lightDir; float cosTheta = dot(ray.direction, sunDir);
	float hgScattering = hgPhaseCloud(cosTheta), hgMultiScat = hgPhase(cosTheta, 0.3f);
	float lightAbsorption = 0.0f, directIntensity = 0.0f, ambientIntensity = 0.0f, distanceSum = 0.0f;
	float stepMul = 3.0f; uint32 missCount = 0; bool fastMarching = true;

	while (tRay.x < tRay.y && lightAbsorption < 1.0f)
	{
		float stepSize = calcStepSize(tRay.x) * stepMul; 
		float3 samplePos = fma(ray.direction, float3(tRay.x), pc.cameraPos);
		float3 cloudData = sampleDataFields(dataFields, pc.cameraPos, samplePos, fieldWindDir);
		float relativeHeight = calcRelativeHeight(pc.bottomRadius, pc.coverage, samplePos, invThickness);
		float vertProfile = calcVerticalProfile(verticalProfile, pc.temperature, cloudData, relativeHeight);
		float cloudCoverage = calcCloudCovergage(pc.coverage, cloudData);
		float dimProfile = vertProfile * cloudCoverage;

		if (dimProfile > 0.0f)
		{
			missCount = 0;
			if (fastMarching) // Starting high resolution ray marching.
			{
				tRay.x -= stepSize; // Go back one step to not miss any high res samples.
				stepMul = 1.0f; fastMarching = false;
				continue;
			}

			float cloudDensity = calcCloudDensity(noiseShape, pc.cameraPos, 
				samplePos, cloudData, dimProfile, noiseWindDir);
			if (cloudDensity < FLOAT_EPS6)
			{
				tRay.x += stepSize;
				continue;
			}

			float occlustion = cloudDensity * (1.0f - lightAbsorption);
			float attenuation = beerLambertCloud(cloudDensity * 10.0f, cosTheta);
			float ambientScattering = pow(1.0f - dimProfile, 0.5f) * attenuation;
			ambientIntensity += ambientScattering * occlustion;
			lightAbsorption += occlustion; distanceSum += tRay.x * occlustion;

			if (sunIntens < FLOAT_EPS6)
			{
				tRay.x += stepSize;
				continue;
			}

			float lightDensity = 0.0f, lightStep = 0.006f;
			for (uint32 step = 0; step < 10; step++) // 256 meters with 10 samples
			{
				samplePos = fma(sunDir, float3(lightStep), samplePos);
				float3 data = sampleDataFields(dataFields, pc.cameraPos, samplePos, fieldWindDir);
				float height = calcRelativeHeight(pc.bottomRadius, pc.coverage, samplePos, invThickness);
				float profile = calcVerticalProfile(verticalProfile, pc.temperature, 
					data, height) * calcCloudCovergage(pc.coverage, data);
				lightDensity = fma(calcCloudDensity(noiseShape, pc.cameraPos, 
					samplePos, data, profile, noiseWindDir), lightStep, lightDensity);
				lightStep += lightStep * 1.3f;
			}

			float lightTransmittance = beerLambertCloud(lightDensity * 100.0f, cosTheta);
			float multiscattering = calcMultiscattering(hgMultiScat, stepSize, cloudData, 
				relativeHeight, cloudCoverage, dimProfile, lightTransmittance);
			float directScattering = fma(lightTransmittance, hgScattering, multiscattering);
			directIntensity += directScattering * occlustion;
		}
		else if (!fastMarching)
		{
			if (missCount++ >= 10)
			{
				fastMarching = true;
				stepMul = 3.0f;
			}
		}

		tRay.x += stepSize;
	}

	if (lightAbsorption < 1.0f)
	{
		// TODO: trace cirrus
	}
	lightAbsorption = min(lightAbsorption, 1.0f);
	
	// TODO: also support coloring by the storm lightnings.

	if (lightAbsorption == 0.0f)
	{
		fb.color = float4(0.0f);
		return;
	}

	worldPos = ray.direction * (distanceSum / lightAbsorption);
	depth = calcDepth(worldPos * 1000.0f, cc.viewProj);
	float4 ap = getAerialPerspLuminance(cameraVolume, fs.texCoords, length(worldPos));

	float3 directLight = cc.sunLight * directIntensity;
	float3 ambientLight = cc.ambientLight * ambientIntensity;
	float3 lightEnergy = lerp(directLight + ambientLight, ap.rgb, ap.a);
	fb.color = float4(lightEnergy, lightAbsorption);
	fb.depth = float4(depth, float3(0.0f));
}