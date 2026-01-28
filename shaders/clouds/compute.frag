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

#include "common/remap.gsl"
#include "common/depth.gsl"
#include "common/constants.gsl"
#include "atmosphere/common.gsl"

spec const float STEP_ADJ_DIST = 16.384f;

pipelineState
{
	faceCulling = off;
}

in noperspective float2 fs.texCoords;
out float4 fb.color;

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
	float coverage;
	float temperature;
} pc;

//**********************************************************************************************************************
float calcErosion(float value, float oldMin) { return saturate((value - oldMin) / (1.0f - oldMin)); }
float3 calcProjSpherePoint(float3 samplePos) { return normalize(samplePos) * pc.bottomRadius; }

float calcStepSize(float distance)
{
	const float nearStepSize = 0.003f, farStepOffset = 0.06f;
	return fma(farStepOffset * distance, STEP_ADJ_DIST, nearStepSize);
}

float calcRelativeHeight(float3 samplePos, float invThickness)
{
	float3 projPos = calcProjSpherePoint(samplePos);
	float relateiveHeight = fma(distance(samplePos, projPos), 
		invThickness, max(pc.coverage * 0.6f, 0.2f));
	return saturate(relateiveHeight);
}
float calcVerticalProfile(float3 cloudData, float relativeHeight)
{
	float topProfile = textureLod(verticalProfile, float2(min(cloudData.z, pc.temperature), relativeHeight), 0.0f).x;
	float bottomProfile = textureLod(verticalProfile, float2(cloudData.x, relativeHeight), 0.0f).y;
	return topProfile * bottomProfile;
}
float calcCloudCovergage(float3 cloudData)
{
	return saturate(remap(cloudData.y, pc.coverage, 1.0f, 0.0f, 1.0f));
}

float calcCloudMipLevel(float3 samplePos, float scale, float offset)
{
	return log2(fma(max(distance(pc.cameraPos, samplePos) + offset, 0.0f), scale, 1.0f));
}
float3 sampleDataFields(float3 samplePos)
{
	const float posScale = 0.05f;
	float mipLod = calcCloudMipLevel(samplePos, 4.0f, -10.0f);
	samplePos += cc.windDir * cc.currentTime * 0.01f;
	#if defined(USE_TRIPLANAR_CLOUDS)
	float3 weights = pow(abs(normalize(samplePos)), float3(4.0f));
	weights /= (weights.x + weights.y + weights.z);
	float3 x = textureLod(dataFields, samplePos.zy * posScale, mipLod).xyz;
	float3 y = textureLod(dataFields, samplePos.xz * posScale, mipLod).xyz;
	float3 z = textureLod(dataFields, samplePos.xy * posScale, mipLod).xyz;
	return x * weights.x + y * weights.y + z * weights.z;
	#else
	return textureLod(dataFields, samplePos.xz * posScale, mipLod).xyz;
	#endif
}

//**********************************************************************************************************************
float calcCloudDensity(float3 samplePos, float3 cloudData, float dimProfile)
{
	float mipLod = calcCloudMipLevel(samplePos, 4.0f, -10.0f);
	samplePos += cc.windDir.yzx * cc.currentTime * -0.001f;
	float4 noise = textureLod(noiseShape, samplePos * 0.5f, mipLod);
	float wispyNoise = lerp(noise.r, noise.g, dimProfile);
	float billowyNoise = lerp(noise.b * 0.3f, noise.a * 0.3f, pow(dimProfile, 0.25f));
	float noiseComposite = lerp(wispyNoise, billowyNoise, cloudData.z);
	return calcErosion(dimProfile, noiseComposite);
}

float hgPhase(float cosTheta, float eccentricity) // Henyey Greenstein
{
	const float g2 = eccentricity * eccentricity;
	float i = inversesqrt((1.0f + g2) - cosTheta * (eccentricity * 2.0f));
	return (i * i * i) * ((1.0f - g2) * M_1_PI4);
}
float hgPhaseCloud(float cosTheta) // Emulates the sun silver lining highlights.
{
	const float eccentricity = 0.6f, silverIntens = 0.7f, silverSpread = 0.1f;
	return max(hgPhase(cosTheta, eccentricity), hgPhase(cosTheta, 0.99f - silverSpread) * silverIntens);
}

float beerLambertCloud(float accumDensity, float cosTheta) 
{
	float transmission = exp(-accumDensity);
	// HG makes far clouds too dark, clouds away from sun dir need extra scattering.
	float modulated = max(transmission, exp(accumDensity * -0.25f) * 0.7f);
	return mix(transmission, modulated, fma(cosTheta, -0.5f, 0.5f));
}
float calcMultiscattering(float hgMultiScat, float stepSize, float3 cloudData, 
	float relativeHeight, float cloudCoverage, float dimProfile, float transmittacne)
{
	const float depthPower = 0.1f, heightPower = 2.0f;
	return hgMultiScat * remapClamp(dimProfile * stepSize * 1000.0f, 0.1f, 1.0f, 0.0f, 1.0f) * 
		pow(cloudCoverage * cloudData.z, 0.25f) * pow(transmittacne, depthPower) * pow(relativeHeight, heightPower);
}

//**********************************************************************************************************************
void main()
{
	Ray ray = Ray(pc.cameraPos, calcViewDirection(fs.texCoords, cc.invViewProj));
	float2 tRay = raycast2(Sphere(float3(0.0f), pc.topRadius), ray);
	if (!isIntersected(tRay)) // No atmosphere intersection so no clouds.
	{
		fb.color = float4(0.0f);
		return;
	}
	tRay.x = max(tRay.x, pc.minDistance); 

	float2 tBottom = raycast2(Sphere(float3(0.0f), pc.bottomRadius), ray);
	if (isIntersected(tBottom)) // Intesecting bottom clouds sphere.
	{
		tRay = tBottom.x < 0.0f ? // Is camera below bottom clouds level?
			float2(max(tRay.x, tBottom.y), tRay.y) : float2(tRay.x, min(tRay.y, tBottom.x)); 
	}

	// TODO: sample depth buffer and update tMax.

	// Skipping ray if whole planet is behind us.
	if (tRay.y <= tRay.x || tRay.x > pc.maxDistance)
	{
		fb.color = float4(0.0f);
		return;
	}

	const float invThickness = 1.0f / (pc.topRadius - pc.bottomRadius);
	float3 sunDir = -cc.lightDir; float cosTheta = dot(ray.direction, sunDir);
	float hgScattering = hgPhaseCloud(cosTheta), hgMultiScat = hgPhase(cosTheta, 0.3f);
	float lightAbsorption = 0.0f, directIntensity = 0.0f, ambientIntensity = 0.0f;
	float stepMul = 3.0f; uint32 missCount = 0; bool fastMarching = true;

	while (tRay.x < tRay.y && lightAbsorption < 1.0f)
	{
		float stepSize = calcStepSize(tRay.x) * stepMul; 
		float3 samplePos = fma(ray.direction, float3(tRay.x), pc.cameraPos);
		float3 cloudData = sampleDataFields(samplePos);
		float relativeHeight = calcRelativeHeight(samplePos, invThickness);
		float verticalProfile = calcVerticalProfile(cloudData, relativeHeight);
		float cloudCoverage = calcCloudCovergage(cloudData);
		float dimProfile = verticalProfile * cloudCoverage;

		if (dimProfile > 0.0f)
		{
			missCount = 0;
			if (fastMarching) // Starting high resolution ray marching.
			{
				tRay.x -= stepSize; // Go back one step to not miss any high res samples.
				stepMul = 1.0f; fastMarching = false;
				continue;
			}

			float cloudDensity = calcCloudDensity(samplePos, cloudData, dimProfile);
			if (cloudDensity < FLOAT_EPS6)
			{
				tRay.x += stepSize;
				continue;
			}

			float lightDensity = 0.0f, lightStep = 0.006f;
			for (uint32 step = 0; step < 10; step++) // 256 meters with 10 samples
			{
				samplePos = fma(sunDir, float3(lightStep), samplePos);
				float3 data = sampleDataFields(samplePos);
				float height = calcRelativeHeight(samplePos, invThickness);
				float profile = calcVerticalProfile(data, height) * calcCloudCovergage(data);
				lightDensity += calcCloudDensity(samplePos, data, profile) * lightStep;
				lightStep += lightStep * 1.3f;
			}

			float lightTransmittance = beerLambertCloud(lightDensity * 100.0f, cosTheta);
			float multiscattering = calcMultiscattering(hgMultiScat, stepSize, cloudData, 
				relativeHeight, cloudCoverage, dimProfile, lightTransmittance);
			float directScattering = fma(lightTransmittance, hgScattering, multiscattering);
			float attenuation = beerLambertCloud(cloudDensity * 10.0f, cosTheta);
			float ambientScattering = pow(1.0f - dimProfile, 0.5f) * attenuation;

			float occlustion = cloudDensity * (1.0f - lightAbsorption);
			directIntensity += directScattering * occlustion;
			ambientIntensity += ambientScattering * occlustion;
			lightAbsorption += occlustion;
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
	lightAbsorption = min(lightAbsorption, 1.0f);

	float3 directLight = cc.sunLight * directIntensity;
	float3 ambientLight = cc.ambientLight * ambientIntensity;
	fb.color = float4(directLight + ambientLight, lightAbsorption);
}