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

#include "common/depth.gsl"

pipelineState
{
	faceCulling = off;
	blending0 = on;
}

in noperspective float2 fs.texCoords;
out float4 fb.shadow;

uniform sampler2D depthBuffer;

uniform sampler2D
{
	addressMode = repeat;
	filter = linear;
} dataFields;
uniform sampler2D
{
	filter = linear;
} verticalProfile;

uniform pushConstants
{
	float4x4 invViewProj;
	float3 cameraPos;
	float bottomRadius;
	float currentTime;
	float coverage;
	float temperature;
} pc;

//**********************************************************************************************************************
void main()
{
	float depth = textureLod(depthBuffer, fs.texCoords, 0.0f).r;
	if (depth == FAR_PLANE_DEPTH)
		discard;

	Ray ray = Ray(pc.cameraPos, calcViewDirection(fs.texCoords, pc.invViewProj));
	float2 tRay = raycast2(Sphere(float3(0.0f), pc.bottomRadius), ray);

	if (!isIntersected(tRay)) // No bottom cloud layer intersection.
		discard;

	float t = tRay.x < tRay.y ? tRay.y : tRay.x;
	float3 samplePos = fma(ray.direction, float3(t), pc.cameraPos);
	float3 fieldWindDir = calcFieldWindDir(cc.windDir, pc.currentTime);
	float3 cloudData = sampleDataFields(dataFields, pc.cameraPos, samplePos, fieldWindDir);
	float vertProfile = calcVerticalProfile(verticalProfile, pc.temperature, cloudData, 0.0f);
	float cloudCoverage = calcCloudCovergage(pc.coverage, cloudData);
	fb.shadow = float4(float3(1.0f), vertProfile * cloudCoverage);
}