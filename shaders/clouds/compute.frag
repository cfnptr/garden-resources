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
uniform sampler2D
{
	addressMode = repeat;
	filter = linear;
} cirrusShape;

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
	float cumulusCoverage;
	float cirrusCoverage;
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
	float2 rayT = raycast2(Sphere(float3(0.0f), pc.topRadius), ray);
	if (!isIntersected(rayT)) // No atmosphere intersection so no clouds.
	{
		fb.color = float4(0.0f); fb.depth = float4(0.0f);
		return;
	}

	float cirrusT = rayT.x < 0.0f ? rayT.y : rayT.x;
	rayT.x = max(rayT.x, pc.minDistance); 

	float2 bottomT = raycast2(Sphere(float3(0.0f), pc.bottomRadius), ray);
	if (isIntersected(bottomT)) // Intesecting bottom clouds sphere.
	{
		rayT = bottomT.x < 0.0f ? // Is camera below bottom clouds level?
			float2(max(rayT.x, bottomT.y), rayT.y) : float2(rayT.x, min(rayT.y, bottomT.x)); 
	}

	float depth = textureLod(hizBuffer, fs.texCoords, 0.0f).r;
	float3 worldPos = calcWorldPosition(depth, fs.texCoords, cc.invViewProj);
	rayT.y = depth > 0.0f ? min(rayT.y, length(worldPos * 0.001f)) : rayT.y;

	// Skipping ray if whole planet is behind us.
	if (rayT.y <= rayT.x || rayT.x > pc.maxDistance)
	{
		fb.color = float4(0.0f); fb.depth = float4(0.0f);
		return;
	}

	float starIntens = dot(cc.starLight, cc.starLight);
	float invThickness = 1.0f / (pc.topRadius - pc.bottomRadius);
	float3 fieldWindDir = calcFieldWindDir(cc.windDir, pc.currentTime);
	float3 shapeWindDir = calcShapeWindDir(cc.windDir, pc.currentTime);
	float3 starDir = -cc.lightDir; float cosTheta = dot(ray.direction, starDir);
	float hgScattering = hgPhaseCloud(cosTheta), hgMultiScat = hgPhase(cosTheta, 0.3f);
	float lightAbsorption = 0.0f, directIntensity = 0.0f, ambientIntensity = 0.0f, distanceSum = 0.0f;
	float stepMul = 3.0f; uint32 missCount = 0; bool fastMarching = true;

	while (rayT.x < rayT.y && lightAbsorption < 1.0f)
	{
		float stepSize = calcStepSize(rayT.x) * stepMul; 
		float3 samplePos = fma(ray.direction, float3(rayT.x), pc.cameraPos);
		float3 cloudData = sampleDataFields(dataFields, pc.cameraPos, samplePos, fieldWindDir, 0.02f);
		float relativeHeight = calcRelativeHeight(pc.bottomRadius, pc.cumulusCoverage, samplePos, invThickness);
		float vertProfile = calcVerticalProfile(verticalProfile, pc.temperature, cloudData, relativeHeight);
		float cloudCoverage = calcCloudCovergage(pc.cumulusCoverage, cloudData);
		float dimProfile = vertProfile * cloudCoverage;

		if (dimProfile > 0.0f)
		{
			missCount = 0;
			if (fastMarching) // Starting high resolution ray marching.
			{
				rayT.x -= stepSize; // Go back one step to not miss any high res samples.
				stepMul = 1.0f; fastMarching = false;
				continue;
			}

			float cloudDensity = calcCloudDensity(noiseShape, pc.cameraPos, 
				samplePos, cloudData, dimProfile, shapeWindDir);
			if (cloudDensity < FLOAT_EPS6)
			{
				rayT.x += stepSize;
				continue;
			}

			const float ambienScattFactor = 5.0f;
			float occlustion = cloudDensity * (1.0f - lightAbsorption);
			float attenuation = beerLambertCloud(cloudDensity * ambienScattFactor, cosTheta);
			float ambientScattering = pow(1.0f - dimProfile, 0.5f) * attenuation;
			ambientIntensity += ambientScattering * occlustion;
			lightAbsorption += occlustion; distanceSum += rayT.x * occlustion;

			if (starIntens < FLOAT_EPS6)
			{
				rayT.x += stepSize;
				continue;
			}

			float lightDensity = 0.0f, lightStep = 0.006f;
			for (uint32 step = 0; step < 10; step++) // 256 meters with 10 samples
			{
				samplePos = fma(starDir, float3(lightStep), samplePos);
				float3 data = sampleDataFields(dataFields, pc.cameraPos, samplePos, fieldWindDir, 0.02f);
				float height = calcRelativeHeight(pc.bottomRadius, pc.cumulusCoverage, samplePos, invThickness);
				float profile = calcVerticalProfile(verticalProfile, pc.temperature, 
					data, height) * calcCloudCovergage(pc.cumulusCoverage, data);
				lightDensity = fma(calcCloudDensity(noiseShape, pc.cameraPos, 
					samplePos, data, profile, shapeWindDir), lightStep, lightDensity);
				lightStep += lightStep * 1.3f;
			}

			const float directScattFactor = 100.0f;
			float lightTransmittance = beerLambertCloud(lightDensity * directScattFactor, cosTheta);
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

		rayT.x += stepSize;
	}

	if (lightAbsorption < 1.0f)
	{
		fieldWindDir = fma(fieldWindDir, float3(0.3f), float3(16.0f)); shapeWindDir *= 2.0f;
		float3 samplePos = fma(ray.direction, float3(cirrusT), pc.cameraPos);
		float3 cloudData = sampleDataFields(dataFields, pc.cameraPos, samplePos, fieldWindDir, 0.025f);
		float3 cirrusShapeData = sampleCirrusShape(cirrusShape, pc.cameraPos, samplePos, shapeWindDir);
		float cloudCoverage = calcCloudCovergage(pc.cirrusCoverage, cloudData);
		float cirrusDensity = calcCirrusDensity(cloudData, cirrusShapeData, cloudCoverage);

		if (cirrusDensity > FLOAT_EPS6)
		{
			const float ambientScattFactor = 4.0f;
			float occlustion = cirrusDensity * (1.0f - lightAbsorption);
			float attenuation = beerLambertCloud(cirrusDensity, cosTheta) * ambientScattFactor;
			float ambientScattering = pow(1.0f - cloudCoverage, 0.5f) * attenuation;
			ambientIntensity += ambientScattering * occlustion;
			lightAbsorption += occlustion; distanceSum += cirrusT * occlustion;

			if (starIntens > FLOAT_EPS6)
			{
				const float lightStep = 25.0f; // Based on the 0.006f
				float lightDensity = 0.0f;
				for (uint32 step = 0; step < 4; step++)
				{
					samplePos = fma(starDir, float3(lightStep), samplePos);
					float3 data = sampleDataFields(dataFields, pc.cameraPos, samplePos, fieldWindDir, 0.025f);
					float3 shapeData = sampleCirrusShape(cirrusShape, pc.cameraPos, samplePos, shapeWindDir);
					float coverage = calcCloudCovergage(pc.cirrusCoverage, data);
					lightDensity += calcCirrusDensity(data, shapeData, coverage);
				}

				const float directScattFactor = 0.25f;
				float lightTransmittance = beerLambertCloud(lightDensity * directScattFactor, cosTheta);
				directIntensity += lightTransmittance * hgScattering * occlustion;
			}
		}
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

	float3 directLight = cc.starLight * directIntensity;
	float3 ambientLight = cc.ambientLight * ambientIntensity;
	float3 lightEnergy = lerp(directLight + ambientLight, ap.rgb, ap.a);
	fb.color = float4(lightEnergy, lightAbsorption);
	fb.depth = float4(depth, float3(0.0f));
}