/*
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Usage:
  const gui = new dat.GUI();
  const ca = new CA(gl, models_json, [W, H], gui); // gui is optional
  ca.step();
  
  ca.paint(x, y, radius, modelIndex);
  ca.clearCircle(x, y, radius;

  const stats = ca.benchmark();
  ca.draw();
  ca.draw(zoom);
*/


// Maps the position which is in [-1.0, 1.0]
// to uv which is in [0, 1]


function defInput(name) {
    return `
        uniform Tensor ${name};
        uniform sampler2D ${name}_tex;

        vec4 ${name}_index_read(float idx, float ch) {return _index_read(${name}, ${name}_tex, idx, ch);}
        vec4 ${name}_read(vec2 pos, float ch) {return _read(${name}, ${name}_tex, pos, ch);}
        vec4 ${name}_read01(vec2 pos, float ch) {return _read01(${name}, ${name}_tex, pos, ch);}
        vec4 ${name}_readUV(vec2 uv) {return _readUV(${name}, ${name}_tex, uv);}
    `
}

const COMMON_PREFIX = `
    // #ifdef GL_FRAGMENT_PRECISION_HIGH
    //     precision highp float;
    //     precision highp sampler2D;
    //     precision highp int;
    // #else
    //     precision mediump float;
    //     precision mediump sampler2D;
    //     precision mediump int;
    // #endif
    
    precision highp float;
    precision highp sampler2D;
    precision highp int;
    // precision highp mat3;
    
    

    // "Hash without Sine" by David Hoskins (https://www.shadertoy.com/view/4djSRW)
    float hash13(vec3 p3) {
      p3  = fract(p3 * .1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }
    vec2 hash23(vec3 p3)
    {
        p3 = fract(p3 * vec3(.1031, .1030, .0973));
        p3 += dot(p3, p3.yzx+33.33);
        return fract((p3.xx+p3.yz)*p3.zy);
    }

    struct Tensor {
        vec2 size;
        vec2 gridSize;
        float depth, depth4;
        
        // tensor.gridSize is the size of the rectangle 
        // containing the channel information of the tensor
        // tensor.size is the spatial size of the original tensor
        // depth: Number of channels
        // depth4: Number of channels // 4
    };
    

    vec4 _readUV(Tensor tensor, sampler2D tex, vec2 uv) {
        highp vec4 v = texture2D(tex, uv);
        return v;
    }
    vec2 _getUV(Tensor tensor, vec2 pos, float ch) {
        // pos is the absolute coordinate
        // uv is the texture coordinate
        
        ch += 0.5;
        // [tx, ty] the offset to move to get the desired channel
        float tx = floor(mod(ch, tensor.gridSize.x));
        float ty = floor(ch / tensor.gridSize.x);
        
        #ifdef OURS
            // vec2 p = clamp(pos / tensor.size, 0.0, 1.0 - 1.0 / tensor.size.y); // replicate padding
            highp vec2 p = clamp(pos, vec2(0.0, 0.0), tensor.size - 0.5); // replicate padding
            p = p / tensor.size;
            // vec2 p = clamp(pos / tensor.size, 0.0, 1.0 - 0.0 / tensor.size.y); // replicate padding
        #else
            highp vec2 p = fract(pos/tensor.size); // circular padding
        #endif 
        
         
        p += vec2(tx, ty); 
        
        p /= tensor.gridSize;
        
        // the output p is in range [0.0, 1.0] 
        
        return p;
    }
    vec4 _read01(Tensor tensor, sampler2D tex, vec2 pos, float ch) {
        // Returns the scaled value of the tensor (between 0.0 and 1.0)
        return texture2D(tex, _getUV(tensor, pos, ch));
    }
    vec4 _read(Tensor tensor, sampler2D tex, vec2 pos, float ch) {
        // Returns the correct value of the tensor
        highp vec2 p = _getUV(tensor, pos, ch);
        return _readUV(tensor, tex, p);
    }
    
    vec4 _index_read(Tensor tensor, sampler2D tex, float idx, float ch) {
        // The indices are basically integer values. So we add 0.5 to make the mod(), and floor() more robust.
        idx += 0.5; 
        vec2 pos = vec2(floor(mod(idx, tensor.size.x)), floor(idx  / tensor.size.x)) + 0.5;
        // Taking floor(mod()) is essential.
        // Returns the correct value of the tensor
        highp vec2 p = _getUV(tensor, pos, ch);
        return _readUV(tensor, tex, p);
    }
    
    ${defInput('u_input')}
    
    vec2 texCoordFromIndex(float index, const vec2 textureSize) {
        index += 0.5; // Add 0.5 to make mod() and floor() more robust
        vec2 colRow = vec2(floor(mod(index, textureSize.x)), floor(index / textureSize.x));
        return vec2((colRow + 0.5) / textureSize);
    }
    
    
    vec4 tanh(vec4 x) {
        vec4 ex = exp(x);
        vec4 emx = exp(-x);
        return (ex - emx) / (ex + emx);
    }
    
    vec3 tanh(vec3 x) {
        vec3 ex = exp(x);
        vec3 emx = exp(-x);
        return (ex - emx) / (ex + emx);
    }
    
    float tanh(float x) {
        float ex = exp(x);
        float emx = exp(-x);
        return (ex - emx) / (ex + emx);
    }
    
  
  
`


const NCA_PREFIX = `
    #ifdef SPARSE_UPDATE
        uniform highp sampler2D u_shuffleTex, u_unshuffleTex;
        uniform highp vec2 u_shuffleOfs;
    #endif

    uniform Tensor u_output;
    vec2 getOutputXY() {
        // gl_FragCoord is the coordinate in the texture
        // which contains the tensor information
        // If the original tensor is 3x3 and has 32 channels then
        // The first channel of the texture would look like this
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 4 4 4 5 5 5 6 6 6 7 7 7 
        // 4 4 4 5 5 5 6 6 6 7 7 7
        // 4 4 4 5 5 5 6 6 6 7 7 7
        
        // Taking the mode with respect to the output size
        // will give us the spatial index in the original tensor

        vec2 xy = mod(gl_FragCoord.xy, u_output.size);  
        
        return xy;
        
    }
    float getOutputChannel() {
        highp vec2 xy = floor(gl_FragCoord.xy/u_output.size);
        return xy.y*u_output.gridSize.x+xy.x;
    }
    
    int integerModulo(const int m, const int n) {
        return m - (m / n) * n;
    }
    
    float getOutputVertexIdx(vec2 u_positionsSize) {
        // int A = int(u_positionsSize.x + 0.5);
        // int B = int(u_positionsSize.y + 0.5);
        //
        // int x = int(gl_FragCoord.x);
        // int y = int(gl_FragCoord.y);
        //
        // x = integerModulo(x, A);
        // y = integerModulo(y, B);
        //
        // float vertexIdx = float(y * A + x);
        
        // gl_FragCoord is centered in the cell so it goes from (0.5, 0.5) to (H - 0.5, W - 0.5)
        // That's why we take floor. 
        // The returned vertexIdx is effectively an integer.
        #ifdef SPARSE_UPDATE
            vec2 xy = mod(gl_FragCoord.xy, u_output.size);
            
            vec2 real_xy = texture2D(u_shuffleTex, xy/u_output.size).xy + u_shuffleOfs;
            xy = floor(mod(real_xy + 0.5, u_input.size));
            float vertexIdx = xy.y * u_positionsSize.x + xy.x;
            return vertexIdx;
        #else    
            vec2 xy = floor(mod(gl_FragCoord.xy, u_output.size));
            float vertexIdx = xy.y * u_positionsSize.x + xy.x;
            return vertexIdx;
        
        #endif
        
    }

    void setOutput(vec4 v) {      
        #ifndef OURS
            v = clamp(v, -2.0, 2.0);
        #else    
            // v = clamp(v, -6.0, 6.0);
        #endif
        gl_FragColor = v;
    }



    

    uniform float u_angle, u_alignment;
    uniform highp vec2 HW;
    
    mat2 rotate(float ang) {
        float s = sin(ang), c = cos(ang);
        return mat2(c, s, -s, c);
    }
    
    mat3 rotateZ(float ang) {
        float s = sin(ang), c = cos(ang);
        return mat3(c, -s, 0, s, c, 0, 0, 0, 1);
    
    }
    
    mat3 rotateY(float ang) {
        float s = sin(ang), c = cos(ang);
        return mat3(c, 0, s, 0, 1,  0, -s, 0, c);
    
    }

    mat3 rotateX(float ang) {
        float s = sin(ang), c = cos(ang);
        return mat3(1, 0, 0, 0, c, -s, 0, s, c);
    
    }
    
    mat3 rotateU(vec3 u, float ang) {
        float s = sin(ang), c = cos(ang);
        u = normalize(u);
        float x = u.x, y = u.y, z = u.z;
        
        return mat3(
            c + x * x * (1.0 - c),        x * y * (1.0 - c) - z * s,        x * z * (1.0 - c) + y * s,
            y * x * (1.0 - c) + z * s,      c + y * y * (1.0 - c),          y * z * (1.0 - c) - x * s,
            z * x * (1.0 - c) - y * s,    z * y * (1.0 - c) + x * s,          c + z * z * (1.0 - c)
        );
    
    
    }
    
    mat3 myTranspose(mat3 m) {
        return mat3(m[0][0], m[1][0], m[2][0],
                    m[0][1], m[1][1], m[2][1],
                    m[0][2], m[1][2], m[2][2]);
    
    }
    
    mat3 myInverse(mat3 m) {
        float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
        float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
        float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];
        float b01 = a22 * a11 - a12 * a21;
        float b11 = -a22 * a10 + a12 * a20;
        float b21 = a21 * a10 - a11 * a20;
        float det = a00 * b01 + a01 * b11 + a02 * b21;
        return mat3(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11),
                    b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10),
                    b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) / det;
    }
    

    vec3 getCellDirection(vec3 pos, vec3 dir) {
        if (u_alignment == 1.0) {
            vec3 u = normalize(pos);
            float x = u.x, y = u.y, z = u.z;
            
            float eps = 1e-6;
            vec3 r_hat = vec3(x, y, z);
            vec3 phi_hat = normalize(vec3(-y, x, 0.0));
            vec3 theta_hat = cross(phi_hat, r_hat);
            
            dir = vec3(dot(r_hat, dir), dot(theta_hat, dir), dot(phi_hat, dir));
            // dir = normalize(dir + eps);
            
        } else if (u_alignment == 2.0) {
            vec3 u = normalize(pos);
            float x = u.x, y = u.y, z = u.z;
            
            float eps = 1e-6;
            vec3 ro_hat = normalize(vec3(x, 0, z));
            vec3 phi_hat = normalize(vec3(-z, 0, x));
            vec3 z_hat = vec3(0.0, 1.0, 0.0);
            
            dir = vec3(dot(ro_hat, dir), dot(phi_hat, dir), dot(z_hat, dir));
            
            
        }
        
        return dir;
    }

`;


const nca_vs_code = `
    attribute highp vec4 position;
    varying highp vec2 uv;
    void main() {
        uv = position.xy*0.5 + 0.5;
        gl_Position = position;
    }
`;

const render_vs_code = `  

  ${defInput('u_graft')}

  uniform float u_bumpiness;


  attribute float a_positionIndex;
  attribute float a_normalIndex;
  attribute float a_uvIndex;

  
  uniform float u_numVertices;
    
  uniform sampler2D u_neighbors;
  uniform vec2 u_neighborsSize;

  uniform sampler2D u_positions;
  uniform vec2 u_positionsSize;
  
  uniform sampler2D u_normals;
  uniform vec2 u_normalsSize;
  
  uniform sampler2D u_uvs;
  uniform vec2 u_uvsSize;

  uniform mat4 u_projection;
  uniform mat4 u_view;
  uniform mat4 u_world;

  varying vec3 v_normal;
  varying vec3 v_position;
  varying vec3 albedo;
  varying vec3 shading_normal;
  varying float height;
  varying float roughness;
  varying float ambient_occlusion;
  varying float graft_weight;
  
  uniform bool u_enable_normal_map;
  uniform bool u_enable_ao_map;
  uniform bool u_enable_roughness_map;
  
  

  int integerModulo(const int m, const int n) {
    return m - (m / n) * n;
  }
  


   

  void main() {
    vec2 ptc = texCoordFromIndex(a_positionIndex, u_positionsSize);
    vec3 position = texture2D(u_positions, ptc).rgb;
    // vec3 position = u_vertexPos_index_read(a_positionIndex, 0.0).rgb;
    
    vec2 ntc = texCoordFromIndex(a_positionIndex, u_normalsSize);
    vec3 surface_normal = normalize(texture2D(u_normals, ntc).xyz);


    vec4 state_set0 = u_input_index_read(a_positionIndex, 0.0);
    vec4 state_set1 = u_input_index_read(a_positionIndex, 1.0);
    vec4 state_set2 = u_input_index_read(a_positionIndex, 2.0);
  
    albedo = state_set0.xyz;
    albedo = albedo + 0.5;
    albedo = clamp(albedo, 0.0, 1.0);
    
    
    
    
    // if (u_expType < 0.5) {
    //     height = state_set1.z; 
    //     height = clamp(height + 0.5, 0.0, 1.0);
    //     height = pow(height, 2.0);
    //     position += surface_normal * height * u_bumpiness;
    // } else {
    //     float delta_x = state_set0.w;
    //     float delta_y = state_set1.x;
    //     float delta_z = state_set1.y;
    //     vec3 delta_p = tanh(vec3(delta_x, delta_y, delta_z)) * 0.2;
    //     height = clamp(length(delta_p), 0.0, 1.0);
    //     position += delta_p * u_bumpiness;
    // }
    
    height = state_set1.z; 
    height = clamp(height + 0.5, 0.0, 1.0);
    height = pow(height, 2.0);
    position += surface_normal * height * u_bumpiness;
    v_position = position;
    
    gl_Position = u_projection * u_view * u_world * vec4(position, 1.0);
    
    // v_normal = (u_view * vec4(surface_normal, 0.0)).xyz;
    // v_normal = normalize(v_normal);
    v_normal = normalize(surface_normal);
    // v_position = (u_view * vec4(position, 1.0)).xyz;
    
    
    if (u_enable_normal_map) {
        shading_normal = clamp(vec3(state_set1.w, state_set2.x, state_set2.y) + 0.5, 0.0, 1.0);
        
    } else {
        shading_normal = vec3(0.5, 0.5, 1.0);
    }
    
    if (u_enable_roughness_map) {
        roughness = clamp(state_set2.z + 0.5, 0.1, 1.0);
    } else {
        roughness = 0.5;
    }
    
    if (u_enable_ao_map) {
        ambient_occlusion = clamp(state_set2.w + 0.5, 0.05, 1.0);
    } else {
        ambient_occlusion = 1.0;
    }
    
    graft_weight = u_graft_index_read(a_positionIndex, 0.0).x;
    
    return;
  }
`;

const render_fs_code = `
    // Some code segments from https://learnopengl.com/PBR/Lighting by Joey de Vries
    // licensed under CC BY-NC 4.0 license as published by Creative Commons
    precision highp float;
    const float INV_4PI2 = 0.02533029591;
    const float INV_PI = 0.31830988618;
    const float PI = 3.14159265359;
    
    vec3 fresnelSchlick(float cosTheta, vec3 F0) {
        return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }  
    
    float DistributionGGX(vec3 N, vec3 H, float roughness) {
        float a = roughness*roughness;
        // float a = roughness;
        float a2     = a*a;
        float NdotH  = max(dot(N, H), 0.0);
        float NdotH2 = NdotH*NdotH;
        float num   = a2;
        float denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = PI * denom * denom;
        return num / denom;
    }

    float GeometrySchlickGGX(float NdotV, float roughness) {
        float r = (roughness + 1.0);
        float k = (r*r) / 8.0;
    
        float num   = NdotV;
        float denom = NdotV * (1.0 - k) + k;
        return num / denom;
    }
    float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float ggx2  = GeometrySchlickGGX(NdotV, roughness);
        float ggx1  = GeometrySchlickGGX(NdotL, roughness);
        return ggx1 * ggx2;
    }
    
    varying vec3 v_normal;
    varying vec3 v_position;
    varying vec3 albedo;
    
    varying vec3 shading_normal;
    varying float roughness;
    varying float height;
    varying float ambient_occlusion;
    varying float graft_weight;

    uniform float u_point_light_strength;
    uniform float u_ambient_light_strength;

    uniform vec3 u_cameraPosition;
    
    uniform float u_visMode;
    const vec3 PHI = vec3(7.00);


    void main () {
        // gl_FragColor = vec4(abs(v_normal), 1.0);
        // return;
        if (u_visMode < 0.5) {
        
        } else if (u_visMode < 1.5) {
            gl_FragColor = vec4(vec3(albedo), 1.0);
            return;
        } else if (u_visMode < 2.5) {
            gl_FragColor = vec4(shading_normal, 1.0);
            return;
        } else if (u_visMode < 3.5) {
            gl_FragColor = vec4(vec3(height), 1.0);
            return;
        } else if (u_visMode < 4.5) {
            gl_FragColor = vec4(vec3(roughness), 1.0);
            return;
        } else if (u_visMode < 5.5) {
            gl_FragColor = vec4(vec3(ambient_occlusion), 1.0);
            return;
        } else if (u_visMode < 6.5) {
            gl_FragColor = vec4(vec3(graft_weight), 1.0);
            return;
        }
    
    
        vec3 normalized_shading_normal = normalize(shading_normal - 0.5 + vec3(0.0, 0.0, 0.0001));
        vec3 t_hat;
        if (v_normal.y != 0.0 || v_normal.z != 0.0) {
            t_hat = normalize(cross(v_normal, vec3(1.0, 0.0, 0.0)));
        } else {
            t_hat = normalize(cross(v_normal, vec3(0.0, 1.0, 0.0)));
        }
        
        vec3 b_hat = normalize(cross(v_normal, t_hat));
        vec3 N = normalize(normalized_shading_normal.x * t_hat
                             + normalized_shading_normal.y * b_hat
                             + normalized_shading_normal.z * v_normal);
        
        vec3 V = normalize(u_cameraPosition - v_position);
        
        vec3 light_position = normalize(u_cameraPosition) * 2.0;  
        vec3 L = normalize(light_position - v_position);
        vec3 H = normalize(V + L);
        
        float distance = length(light_position - v_position);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = PHI * attenuation * u_point_light_strength; 
        
        vec3 Lo = vec3(0.0);
        vec3 F0 = vec3(0.04);
        float metallic = 0.0;
        F0 = mix(F0, albedo, metallic);
        
        
        float NDF = DistributionGGX(N, H, roughness);        
        float G = GeometrySmith(N, V, L, roughness);      
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);       
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;  
            
        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);                
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
        
        vec3 ambient = vec3(0.8) * albedo * ambient_occlusion * u_ambient_light_strength; 
        vec3 output_color = ambient + Lo;
        // output_color = output_color / (output_color + vec3(1.0));
        // output_color = pow(output_color, vec3(1.0/2.2));  
       
        gl_FragColor = vec4(output_color, 1.0);
          
        // vec3 wi = light_position - v_position;
        // float distance_to_light = length(wi);
        // wi = wi / (0.0001 + distance_to_light);
        //
        // vec3 wo = normalize(u_cameraPosition - v_position);
        //
        // float cos_th = dot(wi, n);
        //
        // vec3 fr = brdf(n, wi, wo, albedo, 1.0);
        // float ambient = 0.25 * ambient_occlusion;
        // vec3 point_light_contribution = fr * max(0.0, cos_th) * PHI * INV_4PI2 / (distance_to_light * distance_to_light);
        // vec3 output_color = point_light_contribution + ambient * albedo;   
        // output_color = clamp(output_color, 0.0, 1.0);     
        //
        // gl_FragColor = vec4(output_color, 1.0);
        
        //
        //
        // float cos_th = dot(light_dir, n);
        // float diffuse = clamp(cos_th, 0.0, 1.0);
        //
        //
        // // Phong Shading
        // vec3 V = normalize(u_cameraPosition - v_position);
        // vec3 R = normalize(2.0 * cos_th * n - light_dir);
        // // vec3 H = (V + light_dir) / 2.0;
        // // H = normalize(H);
        // // float specular = pow(clamp(dot(H, n), 0.0, 1.0), 4.0);
        //
        // // Simplified Phong Shading
        // // float specular = pow(clamp(cos_th, 0.0, 1.0), 3.0);
        //
        // float specularity = 1.0 / (roughness + 0.01);
        // float specular = pow(clamp(dot(V, R), 0.0, 1.0), specularity);
        // // specular = specular * (1.0 - roughness);
        //
        // diffuse = 1.0 - specular;
        // // diffuse = clamp(roughness - specular, 0.0, 1.0);
        // specular = specular * 2.5 / (distance_to_light * distance_to_light);
        // diffuse = diffuse * 2.5 / (distance_to_light * distance_to_light);
        // float ambient = 0.25 * ambient_occlusion;
        //
        // vec3 output_color = clamp(color, 0.0, 1.0);
        // output_color = output_color * (diffuse + ambient) + specular;   
        // output_color = clamp(output_color, 0.0, 1.0);     
        //
        // gl_FragColor = vec4(output_color, 1.0);
    }
  `;


const NCA_PROGRAMS = {
    paint: `
    uniform highp vec2 u_pos;
    uniform vec2 u_canvas_size;
    uniform float u_r;
    uniform highp vec4 u_brush;
    
    uniform bool u_reset;
    uniform bool u_residual;
    
    
    uniform sampler2D u_positions;
    uniform vec2 u_positionsSize;
    
    uniform sampler2D u_normals;
    uniform vec2 u_normalsSize;
    
    uniform mat4 u_projection;
    uniform mat4 u_view;
    uniform mat4 u_world;

    void main() {
        // float vertexIdx = getOutputVertexIdx(u_input.size);
        vec2 vertex_xy = floor(mod(gl_FragCoord.xy, u_output.size));
        float vertexIdx = vertex_xy.y * u_input.size.x + vertex_xy.x;
            
        vec2 ptc = texCoordFromIndex(vertexIdx, u_positionsSize);
        vec3 position = texture2D(u_positions, ptc).rgb;
        
        vec2 ntc = texCoordFromIndex(vertexIdx, u_normalsSize);
        vec3 normal = normalize(texture2D(u_normals, ntc).xyz);
        
        float delta_u = u_input_read(getOutputXY(), 0.0).w;
        vec3 new_position = position + normal * delta_u * 0.0;
        vec4 pos_on_screen = u_projection * u_view * u_world * vec4(new_position, 1.0);
        pos_on_screen = pos_on_screen / pos_on_screen.w;
        pos_on_screen.y *= -1.0;
        
        vec2 xy = u_pos;
        vec2 xy_out = (pos_on_screen.xy + 1.0) / 2.0; 
        // vec2 xy_out = u_canvas_size * (pos_on_screen.xy);
        
        if (u_reset) {
            setOutput(u_brush);
            return;  
        }  
              
        if (length(abs(xy_out-xy))>=u_r) {
            if (u_residual) {
                setOutput(u_input_read(getOutputXY(), getOutputChannel()));
                return;
            } else {
                discard;
            }
        
        }
          
          
        // vec3 surface_normal = texture2D(u_normals, texCoordFromIndex(vertexIdx, u_normalsSize)).xyz;
        vec3 v_normal = (u_view * vec4(normal, 0.0)).xyz;

        if (v_normal.z < 0.0) {
            if (u_residual) {
                setOutput(u_input_read(getOutputXY(), getOutputChannel()));
                return;
            } else {
                discard;
            }

        }

        
        if (u_residual) {
            float brush_strength = pow(v_normal.z, 6.0);
            vec2 xy = getOutputXY();
            vec4 result = u_brush * brush_strength + u_input_read(xy, getOutputChannel());
            result = clamp(result, 0.0, 1.0);
            setOutput(result);
        } else {
            setOutput(u_brush);
        }

        

    }`,
    dense: `    
    //u_weightTex contains the layer weights
    uniform sampler2D u_weightTex;
    uniform float u_seed, u_fuzz, u_updateProbability;
    uniform highp vec2 u_weightCoefs; // scale, center
    uniform highp vec2 u_layout;
    uniform highp vec2 grid_size;
    uniform bool bias, pos_emb, relu;
    
    const float MAX_PACKED_DEPTH = 50.0;
    
    uniform float u_texture_idx;
    
    ${defInput('u_graft_weights')}
    uniform bool u_enable_graft;
    
    
    
    vec4 readWeightUnscaled(vec2 p) {
        highp vec4 w = texture2D(u_weightTex, p);
        return w-u_weightCoefs.y; // centerize
    }
    
    void main() {
      vec2 xy = getOutputXY();
      
      
      #ifndef SPARSE_UPDATE
      if (hash13(vec3(xy, u_seed)) > u_updateProbability) {
        return;
      }
      // xy = mod(gl_FragCoord.xy, u_input.size);  
      #endif
      
      
      vec2 realXY = xy;
      #ifdef SPARSE_UPDATE
        realXY = texture2D(u_shuffleTex, xy/u_output.size).xy + u_shuffleOfs;
        realXY = mod(realXY + 0.5, u_graft_weights.size);
      #endif
      
      if (u_enable_graft) {
        float graft_weight = u_graft_weights_read(realXY, 0.0).x;
        if (graft_weight < MIN_GRAFT_WEIGHT) {
            setOutput(vec4(0.0));
            return;
        } 
      }
      
      
      
      float ch = getOutputChannel();
      if (ch >= u_output.depth4)
          return;


      float d = u_input.depth + 1.0;
      if (pos_emb) {
        d = d + 2.0;
      }
      float dy = 1.0 / (d) / u_layout.y;
      // float dy = 1.0/(d)/u_layout.y;
      // float dy = 1.0/(u_input.depth+1.0)/u_layout.y;
      vec2 p = vec2((ch+0.5)/u_output.depth4, dy*0.5);
      vec2 fuzz = (hash23(vec3(xy, u_seed+ch))-0.5)*u_fuzz;
      // vec2 fuzz = vec2(0.0, 0.0);

      float texture_idx = u_texture_idx + 0.5;

      p.x += floor(mod(texture_idx, u_layout.x));
      p.y += floor(texture_idx/u_layout.x);
      p /= u_layout;
      highp vec4 result = vec4(0.0);
      for (float i=0.0; i < MAX_PACKED_DEPTH; i+=1.0) {
          highp vec4 inVec = u_input_read(xy, i);
          result += inVec.x * readWeightUnscaled(p); p.y += dy;
          result += inVec.y * readWeightUnscaled(p); p.y += dy;
          result += inVec.z * readWeightUnscaled(p); p.y += dy;
          result += inVec.w * readWeightUnscaled(p); p.y += dy;
          if (i+1.5>u_input.depth4) {
              break;
          }
      }
      if (pos_emb) {
        
        highp vec2 pos = floor(realXY + 0.5);
        highp vec2 delta = vec2(0.5, 0.5) / HW;
        highp vec2 pemb = pos / HW;
        pemb = 2.0 * (pemb - 0.5 + delta);
        pemb = rotate(-u_angle) * pemb;
        result += pemb.y * readWeightUnscaled(p); p.y += dy;
        result += pemb.x * readWeightUnscaled(p); p.y += dy;
      
      };
      if (bias) {
        result += readWeightUnscaled(p);  // bias
        // p.y += dy; 
      };
      
      result = result*u_weightCoefs.x;
      if (relu) {
        result = max(result, 0.0);
      
      }
      setOutput(result);
    }`,


    mesh_state_update: `
    ${defInput('u_update')}
    
    ${defInput('u_update_graft')}
    ${defInput('u_graft_weights')}
    
    uniform bool u_enable_graft;
    
    uniform float u_seed, u_updateProbability;
    uniform float u_rate;

    varying vec2 uv;
    
    uniform float u_numVertices;
    uniform sampler2D u_positions;
    uniform vec2 u_positionsSize;


    uniform bool u_hardClamp;
    
    void main() {
      vec2 xy = getOutputXY();
    //   if (xy.y>100.0 && xy.y < 150.0) {
    //       xy.x -= 1.0;
    //   }
      float ch = getOutputChannel();
      vec4 state = u_input_read(xy, ch); //u_input_readUV(uv);
      vec4 update = vec4(0.0);
      vec4 graft_update = vec4(0.0);
      float graft_weight;
      bool enable_grafting = u_enable_graft;
      if (u_enable_graft) {
        graft_weight = u_graft_weights_read(xy, 0.0).x;
        enable_grafting = (graft_weight >= MIN_GRAFT_WEIGHT); 
      }
      
       
      
      #ifdef SPARSE_UPDATE
        vec4 shuffleInfo = texture2D(u_unshuffleTex, fract((xy-u_shuffleOfs)/u_output.size));
        if (shuffleInfo.z > 0.5) {
            // update = u_update_read(shuffleInfo.xy*255.0+0.5, getOutputChannel());
            update = u_update_read(shuffleInfo.xy + 0.5, getOutputChannel());
            if (enable_grafting) {
                graft_update = u_update_graft_read(shuffleInfo.xy + 0.5, getOutputChannel());
            }
            
            // update = u_update_read(shuffleInfo.xy*(HW.x - 1.0)+0.5, getOutputChannel());
        }
      #else
        if (hash13(vec3(xy, u_seed)) <= u_updateProbability) {
            // update = u_update_readUV(uv);    
            update = u_update_read(xy, getOutputChannel());
            if (enable_grafting) {
                graft_update = u_update_graft_read(xy, getOutputChannel());
            }
                
        }          

      #endif
      
      vec4 new_state;
      update = update * u_rate;
      graft_update = graft_update * u_rate;
      if (enable_grafting) {
        new_state = state + mix(update, graft_update, graft_weight);
      } else {
        new_state = state + update;
      }
      
      
      if (u_hardClamp) {
          new_state = clamp(new_state, -1.0, 1.0);
      }
      
      
      setOutput(new_state);
    }`,
    mesh_perception: `
    #define RSH1_SOBEL 0.48860251190292
    #define RSH1_LAP 0.282094791773878
    
    
    ${defInput('u_vertexPos')}
    
    uniform float u_numVertices;
    
    uniform sampler2D u_neighbors;
    uniform vec2 u_neighborsSize;
    
    uniform sampler2D u_positions;
    uniform vec2 u_positionsSize;
    
    uniform sampler2D u_kernels;
    uniform vec2 u_kernelsSize;
    
    uniform sampler2D u_normals;
    uniform vec2 u_normalsSize;
    
    // uniform float u_angle;
    
    uniform float u_seed, u_updateProbability;
    uniform float u_scale;

    vec4 getNeighbors(const float index, const float offset) {
        float scaled_index = index * Q + offset + 0.5; // Add 0.5 to make mod() and floor() more robust
        vec2 colRow = vec2(floor(mod(scaled_index, u_neighborsSize.x)), floor(scaled_index / u_neighborsSize.x));
        vec2 uv = vec2((colRow + 0.5) / u_neighborsSize);
        return texture2D(u_neighbors, uv);
    }  
    
    void main () {
        #ifndef SPARSE_UPDATE
            if (hash13(vec3(getOutputXY(), u_seed)) > u_updateProbability) {
                return;
            }
        
        #endif

        
    
    
        // float vertexIdx = getOutputVertexIdx(u_vertexPos.size);
        float vertexIdx = getOutputVertexIdx(u_positionsSize);
        // setOutput(vec4(vertexIdx / u_numVertices));
        // return;
        
        
        if (vertexIdx > u_numVertices) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
        
        
        // TODO: Optimize by calculating the neighbors only when needed
        float neighbor_indices[MAX_NEIGHBORS];
        int num_neighbors = 0;    
        vec4 last_neighbors;
        
        vec3 average_neighbor_pos = vec3(0.0);
        for (int j = 0; j < MAX_NEIGHBORS; j+=1) {
            if (integerModulo(j, 4) == 0)  {
                float offset = floor((float(j) + 0.5) / 4.0); // without 0.1 it doesn't work. Invesitage why.
                last_neighbors = getNeighbors(vertexIdx, offset);
            }
            float neighbor_idx = last_neighbors[integerModulo(j, 4)];
            if (neighbor_idx < -0.5) {
                break;
            }  else {
                neighbor_indices[j] = neighbor_idx;
                num_neighbors += 1;
                
                average_neighbor_pos += texture2D(u_positions, texCoordFromIndex(neighbor_idx, u_positionsSize)).xyz;
            }
        }
        average_neighbor_pos = average_neighbor_pos / float(num_neighbors);
        
        
        
        // setOutput(vec4(float(num_neighbors == 5)));
        // return;
        

        float ch = getOutputChannel();
        if (ch >= u_output.depth4)
            return;

        float num_perceptions = floor((u_output.depth4 + 0.5) / u_input.depth4); // 5
        float channels_per_filter = floor((u_output.depth4 + 0.5) / num_perceptions); // 3
        
        float filterBand = floor((ch+0.5) / channels_per_filter);
        // inputCh: this is the channel idx in the original tensor
        float inputCh = ch - filterBand * channels_per_filter; 

        float initial_band = 0.5;
        if (filterBand < initial_band) {
            // Laplacian
            vec4 res = -u_input_index_read(vertexIdx, inputCh) * float(num_neighbors);
            for (int j = 0; j < MAX_NEIGHBORS; j+=1) {
                if (j >= num_neighbors) {
                    break;
                }

                res += u_input_index_read(neighbor_indices[j], inputCh);

            }
            // res = res * RSH1_LAP;
            res = res / (u_scale * u_scale);
            res = res * 6.0 / float(num_neighbors);
            setOutput(res);
        }

        else if (filterBand < initial_band + 3.0) {
            //Spherical Harmonics
            vec4 res = vec4(0.0);
            vec4 center_feature = u_input_index_read(vertexIdx, inputCh);
            vec3 center_pos = texture2D(u_positions, texCoordFromIndex(vertexIdx, u_positionsSize)).xyz;
            // vec3 center_pos = u_vertexPos_index_read(vertexIdx, 0.0).xyz;
            vec3 surface_normal = texture2D(u_normals, texCoordFromIndex(vertexIdx, u_normalsSize)).xyz;
            mat3 rotation_matrix = rotateU(surface_normal, u_angle);
            
            vec3 vertex_kernel_row1 = texture2D(u_kernels, texCoordFromIndex(vertexIdx * 3.0, u_kernelsSize)).xyz;
            vec3 vertex_kernel_row2 = texture2D(u_kernels, texCoordFromIndex(vertexIdx * 3.0 + 1.0, u_kernelsSize)).xyz;
            vec3 vertex_kernel_row3 = texture2D(u_kernels, texCoordFromIndex(vertexIdx * 3.0 + 2.0, u_kernelsSize)).xyz;
            
            float gamma = 0.01;
            
            // mat3 vertex_kernel = mat3(
            //                     vertex_kernel_row1 + vec3(1.0, 0.0, 0.0) * gamma,
            //                     vertex_kernel_row2 + vec3(0.0, 1.0, 0.0) * gamma,
            //                     vertex_kernel_row3 + vec3(0.0, 0.0, 1.0) * gamma);
                                
            mat3 vertex_kernel = mat3(vertex_kernel_row1, vertex_kernel_row2, vertex_kernel_row3);
            
            
            mat3 kernel = myTranspose(rotation_matrix) * vertex_kernel;
            // mat3 kernel = myTranspose(rotation_matrix) * myInverse(vertex_kernel);
            // myTranspose(rotation_matrix);
            
            for (int j = 0; j < MAX_NEIGHBORS; j+=1) {
                if (j >= num_neighbors) {
                    break;
                }
                float neighbor_index = neighbor_indices[j];
                vec3 neighbor_pos = texture2D(u_positions, texCoordFromIndex(neighbor_index, u_positionsSize)).xyz;
                // vec3 neighbor_pos = u_vertexPos_index_read(neighbor_index, 0.0).xyz;
                // direction = rotateZ(u_angle) * direction;
                
                vec3 direction = neighbor_pos - center_pos;
                
                // direction = kernel * rotation_matrix * normalize(direction);
                // direction = myInverse(vertex_kernel) * normalize(direction);
                direction =  rotation_matrix * normalize(direction);

                // vec3 direction =  rotation_matrix * normalize(neighbor_pos - average_neighbor_pos);
                // direction = getCellDirection(center_pos, direction);
                
                
                
                float coeff;
                if (filterBand < initial_band + 1.0) {
                    coeff = direction.y;
                } else if (filterBand < initial_band + 2.0) {
                    coeff = direction.z;
                } else {
                    coeff = direction.x;
                }
                
                // res += vec4(neighbor_pos, 1.0);
                
                // coeff *= RSH1_SOBEL;
                res += coeff * (u_input_index_read(neighbor_index, inputCh) - center_feature);
                

            }
            // res += vec4(center_pos, 1.0);
            // res /= (1.0 + float(num_neighbors));
            res = res / u_scale;
            res = res * 6.0 / float(num_neighbors);
            setOutput(res);
            
        } else {
            // Identityanywa
            
            // setOutput(u_input_read(xy, inputCh));
            setOutput(u_input_index_read(vertexIdx, inputCh));
        }
        
    }
    `
}

function createNCAPrograms(gl, defines) {
    defines = defines || '';
    const res = {};
    for (const name in NCA_PROGRAMS) {
        const fs_code = defines + COMMON_PREFIX + NCA_PREFIX + NCA_PROGRAMS[name];
        // vs : vertex shader
        // fs: fragment shader
        const progInfo = twgl.createProgramInfo(gl, [nca_vs_code, fs_code]);
        progInfo.name = name;
        res[name] = progInfo;
    }
    // res is a dictionary of shader programs
    return res;
}

function createRenderPrograms(gl, defines) {
    defines = defines || '';
    const res = {};
    const fs_code = defines + render_fs_code;
    const vs_code = defines + COMMON_PREFIX + render_vs_code;
    // const fs_code = render_fs_code;

    const progInfo = twgl.createProgramInfo(gl, [vs_code, fs_code]);

    return progInfo;

}

function createTensor(gl, w, h, depth, is_float = false) {
    // Pack the depth dimension into the spatial dimension
    const depth4 = Math.ceil(depth / 4);

    let gridW = Math.ceil(Math.sqrt(depth4));
    let gridH = Math.floor((depth4 + gridW - 1) / gridW);


    const texW = w * gridW, texH = h * gridH;
    // const ext = gl.getExtension('EXT_color_buffer_float');


    const attachments = [{
        minMag: gl.NEAREST,
        format: gl.RGBA,
        internalFormat: gl.RGBA32F
    }];
    if (is_float) {
        attachments[0].type = gl.FLOAT;
    }

    const fbi = twgl.createFramebufferInfo(gl, attachments, texW, texH);
    const tex = fbi.attachments[0];
    return {
        _type: 'tensor',
        fbi, w, h, depth, gridW, gridH, depth4, tex,
    };
}

function setTensorUniforms(uniforms, name, tensor) {
    uniforms[name + '.size'] = [tensor.w, tensor.h];
    uniforms[name + '.gridSize'] = [tensor.gridW, tensor.gridH];
    uniforms[name + '.depth'] = tensor.depth;
    uniforms[name + '.depth4'] = tensor.depth4;
    if (name != 'u_output') {
        uniforms[name + '_tex'] = tensor.tex;
    }
}

function createDenseInfo(gl, params) {
    // params is basically one of layers from the json file

    const center = "center" in params ? params.center : 127.0 / 255.0;

    const [in_n, out_n] = params.shape;
    const info = {
        layout: params.layout, out_n,
        // quantScaleZero: params.quant_scale_zero,
        ready: false
    };

    info.pos_emb = params.pos_emb ? "pos_emb" in params : false;
    info.bias = params.bias ? "bias" in params : true;
    var ch_in = in_n;
    ch_in = info.pos_emb ? ch_in - 2 : ch_in;
    ch_in = info.bias ? ch_in - 1 : ch_in;
    info.in_n = ch_in;
    info.coefs = [params.scale, center];

    if ("data_flatten" in params) {
        let width = params.data_shape[1];
        let height = params.data_shape[0];
        info.tex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, src: params.data_flatten, flipY: false, premultiplyAlpha: false,
            width: width, height: height,
            internalFormat: gl.RGBA32F,
            format: gl.RGBA,
            type: gl.FLOAT,
        })
        info.ready = true; // important

    } else {
        info.tex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, src: params.data, flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        }, () => {
            info.ready = true;
        });
    }
    return info;
}

function degToRad(deg) {
    return deg * Math.PI / 180;
}

export class MeshNCA {
    constructor(gl, models, gridSize, gui, our_version = true, mesh = null, texture_idx = 0, graft_idx = 1) {
        // models is basically the json file

        this.gl = gl;


        this.texture_idx = texture_idx;
        this.graft_idx = graft_idx;
        this.enable_grafting = true;

        this.gridSize = gridSize || [96, 96];

        this.updateProbability = 0.5;
        this.shuffledMode = true; // changed
        // alert(this.shuffledMode)

        this.rotationAngle = 0.0;
        this.alignment = 0;
        this.bumpiness = 0.0;
        this.fuzz = 8.0;
        this.perceptionCircle = 0.0;
        this.arrowsCoef = 0.0;
        this.visMode = 'color';
        this.hardClamp = false;

        this.rate = 1.0;
        this.scale = 1.0;

        this.our_version = our_version;
        this.mesh = mesh;
        this.MAX_NEIGHBORS = mesh.MAX_NEIGHBORS; //should be a multiplier of 4

        if (this.mesh) {
            this.setupMesh(this.gl);
        }


        this.layers = [];
        this.setWeights(models);

        // const defs = ('#define MAX_NEIGHBORS ' + toString(this.MAX_NEIGHBORS) + '\n') + (this.our_version ? '#define OURS\n' : '') + (this.shuffledMode ? '#define SPARSE_UPDATE\n' : '');
        const defs = ('#define MAX_NEIGHBORS ' + this.MAX_NEIGHBORS +
            ' \n  #define Q ' + (this.MAX_NEIGHBORS / 4.0).toFixed(1) +
            ' \n' + (this.our_version ? '#define OURS\n' : '')
            + (this.shuffledMode ? '#define SPARSE_UPDATE\n' : '') +
            ' \n #define MIN_GRAFT_WEIGHT 0.0001 \n'
        );


        this.render_progs = createRenderPrograms(gl, defs);
        this.nca_progs = createNCAPrograms(gl, defs);


        // representing vertices of a square with two triangles
        this.quad = twgl.createBufferInfoFromArrays(gl, {
            position: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
        });

        this.setupBuffers();
        const visNames = Object.getOwnPropertyNames(this.buf);
        visNames.push('color');


        if (gui) {
            gui.add(this, 'rotationAngle').min(0.0).max(360.0);
            gui.add(this, 'alignment', {cartesian: 0, polar: 1, bipolar: 2}).listen();
            //gui.add(this, 'fuzz').min(0.0).max(128.0);
            //gui.add(this, 'perceptionCircle').min(0.0).max(1.0);
            //gui.add(this, 'visMode', visNames);

            gui.add(this, 'benchmark');

            // this.benchmark = ()=>{
            //   document.getElementById('log').insertAdjacentHTML('afterbegin', this.benchmark());
            // }


        }

        this.clearCircle(0, 0, 10000);
    }

    setupMesh(gl) {
        // this.meshAttributes = this.mesh.getAttributeTextures(gl, this.MAX_NEIGHBORS);
        this.meshAttributes = this.mesh.meshAttributeTextures;
        this.meshAttributeTextures = twgl.createTextures(gl, this.meshAttributes);


        this.meshBufferInfo = twgl.createBufferInfoFromArrays(gl, {
            a_positionIndex: {size: 1, data: this.mesh.vertexPositionIndices},
            a_normalIndex: {size: 1, data: this.mesh.vertexFaceNormalIndices},
            a_uvIndex: {size: 1, data: this.mesh.vertexUVIndices},
        });

        const obj_extents = this.mesh.getExtents();

        const obj_range = m4.subtractVectors(obj_extents.max, obj_extents.min);
        // amount to move the object so its center is at the origin
        this.objOffset = [0.0, 0.0, 0.0];

        let radius = 2.2;
        this.camera = {
            cameraTarget: [0, 0, 0],
            radius: radius,
            camera_radius: radius,
            cameraPosition: m4.scaleVector(m4.normalize([1, 1, 1]), radius),
            zNear: radius / 100,
            zFar: radius * 3.0,

            fieldOfViewRadians: degToRad(60),
            up: [0, 1, 0],
            aspect: gl.canvas.clientWidth / gl.canvas.clientHeight,

            world: m4.yRotation(0.0),
        };

        // u_world = m4.translate(u_world, ...this.objOffset);


        this.camera.projection = m4.perspective(this.camera.fieldOfViewRadians, this.camera.aspect, this.camera.zNear, this.camera.zFar);
        this.camera.view = m4.inverse(m4.lookAt(this.camera.cameraPosition, this.camera.cameraTarget, this.camera.up));

    }

    recompute_view_matrix() {
        this.camera.view = m4.inverse(m4.lookAt(this.camera.cameraPosition, this.camera.cameraTarget, this.camera.up));
        this.camera.projection = m4.perspective(this.camera.fieldOfViewRadians, this.camera.aspect, this.camera.zNear, this.camera.zFar);
    }

    setupBuffers() {
        const gl = this.gl;

        let [width, height] = [this.meshAttributes.positions.width, this.meshAttributes.positions.height];

        const shuffleH = Math.ceil(height * this.updateProbability);
        const shuffleCellN = shuffleH * width;
        const totalCellN = width * height;
        // const shuffleBuf = new Uint8Array(shuffleCellN * 4); // Indices of the cells to be updated
        // const unshuffleBuf = new Uint8Array(totalCellN * 4);

        const shuffleBuf = new Float32Array(shuffleCellN * 4); // Indices of the cells to be updated
        const unshuffleBuf = new Float32Array(totalCellN * 4);

        let k = 0;


        for (let i = 0; i < totalCellN; ++i) {
            // This exactly updates shuffleCellN of cells.
            if (Math.random() < (shuffleCellN - k) / (totalCellN - i)) {
                shuffleBuf[k * 4 + 0] = i % width;
                shuffleBuf[k * 4 + 1] = Math.floor(i / width);
                unshuffleBuf[i * 4 + 0] = k % width;
                unshuffleBuf[i * 4 + 1] = Math.floor(k / width);
                unshuffleBuf[i * 4 + 2] = 255;
                k += 1;
            }
        }
        this.shuffleTex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, width: width, height: shuffleH, src: shuffleBuf,
            flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        });
        this.unshuffleTex = twgl.createTexture(gl, {
            minMag: gl.NEAREST,
            width: width,
            height: height,
            src: unshuffleBuf,
            flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        });
        this.shuffleOfs = [0, 0];

        const updateH = this.shuffledMode ? shuffleH : height;
        const perception_n = this.layers[0].in_n;
        const lastLayer = this.layers[this.layers.length - 1];
        const channel_n = lastLayer.out_n;

        this.buf = {
            graftWeights: createTensor(gl, width, height, 4, true),
            NewgraftWeights: createTensor(gl, width, height, 4, true),

            meshState: createTensor(gl, width, height, channel_n, true),
            meshNewState: createTensor(gl, width, height, channel_n, true),
            meshPerception: createTensor(gl, width, updateH, perception_n, true),

        };

        this.mesh_init = false;


        for (let i = 0; i < this.layers.length; ++i) {
            const layer = this.layers[i];
            this.buf[`mesh_layer${i}`] = createTensor(gl, width, updateH, layer.out_n, true);
            this.buf[`mesh_layer_graft${i}`] = createTensor(gl, width, updateH, layer.out_n, true);
        }
    }

    step(stage) {
        stage = stage || 'all';
        if (!this.layers.every(l => l.ready))
            return;

//        if (!this.mesh_init) {
//            this.initVertexPosition();
//            this.mesh_init = true;
//        }

        const seed = Math.random() * 1000;

        if (stage == 'all') {
            const [width, height] = [this.meshAttributes.positions.width, this.meshAttributes.positions.height];
            // Random point on the grid
            this.shuffleOfs = [Math.floor(Math.random() * width), Math.floor(Math.random() * height)];
        }

        if (stage == 'all' || stage == 'Perception') {

            this.runLayer(this.nca_progs.mesh_perception, this.buf.meshPerception, {
                u_input: this.buf.meshState,
                u_seed: seed,
                u_updateProbability: this.updateProbability,

                u_neighbors: this.meshAttributeTextures.neighborhood,
                u_neighborsSize: [this.meshAttributes.neighborhood.width, this.meshAttributes.neighborhood.height],

                u_positions: this.meshAttributeTextures.positions,
                u_positionsSize: [this.meshAttributes.positions.width, this.meshAttributes.positions.height],

                u_kernels: this.meshAttributeTextures.vertex_kernels,
                u_kernelsSize: [this.meshAttributes.vertex_kernels.width, this.meshAttributes.vertex_kernels.height],

                u_normals: this.meshAttributeTextures.normals,
                u_normalsSize: [this.meshAttributes.normals.width, this.meshAttributes.normals.height],


                u_numVertices: this.mesh.numVertices,
                u_angle: this.rotationAngle / 180.0 * Math.PI,
                u_scale: this.scale,
            });


        }


        let mesh_inputBuf = this.buf.meshPerception;
        let mesh_inputBuf_graft = this.buf.meshPerception;

        for (let i = 0; i < this.layers.length; ++i) {
            if (stage == 'all' || stage == `FC Layer${i + 1}`)
                var relu = i == 0 ? true : false;

            this.runDense(this.buf[`mesh_layer${i}`], mesh_inputBuf, this.layers[i], this.texture_idx, relu, seed);

            if (this.enable_grafting) {
                this.runDense(this.buf[`mesh_layer_graft${i}`], mesh_inputBuf_graft, this.layers[i], this.graft_idx, relu, seed, true);
                mesh_inputBuf_graft = this.buf[`mesh_layer_graft${i}`];
            }


            mesh_inputBuf = this.buf[`mesh_layer${i}`];
        }
        if (stage == 'all' || stage == 'Stochastic Update') {

            let input_dict = {
                u_input: this.buf.meshState, u_update: mesh_inputBuf,
                u_unshuffleTex: this.unshuffleTex,
                u_seed: seed, u_updateProbability: this.updateProbability,
                u_hardClamp: this.hardClamp,

                u_positions: this.meshAttributeTextures.positions,
                u_positionsSize: [this.meshAttributes.positions.width, this.meshAttributes.positions.height],
                u_numVertices: this.mesh.numVertices,

                u_enable_graft: this.enable_grafting,

                u_rate: this.rate,
            }

            if (this.enable_grafting) {
                input_dict.u_update_graft = mesh_inputBuf_graft;
                input_dict.u_graft_weights = this.buf.graftWeights;
            }

            this.runLayer(this.nca_progs.mesh_state_update, this.buf.meshNewState, input_dict);

        }

        if (stage == 'all') {
            [this.buf.meshState, this.buf.meshNewState] = [this.buf.meshNewState, this.buf.meshState];

        }
    }

    benchmark() {
        const gl = this.gl;
        // const flushBuf = new Uint8Array(4);
        const flushBuf = new Float32Array(4);
        const flush = buf => {
            buf = buf || this.buf.meshState;
            // gl.flush/finish don't seem to do anything, so reading a single
            // pixel from the state buffer to flush the GPU command pipeline
            twgl.bindFramebufferInfo(gl, buf.fbi);
            gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, flushBuf);
        }

        // const flushBuf = new Uint8Array(4);
        // const flush = buf=>{
        //     buf = buf || this.buf.state;
        //     // gl.flush/finish don't seem to do anything, so reading a single
        //     // pixel from the state buffer to flush the GPU command pipeline
        //     twgl.bindFramebufferInfo(gl, buf.fbi);
        //     gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, flushBuf);
        // }

        flush();
        const stepN = 500;
        const start = Date.now();
        for (let i = 0; i < stepN; ++i)
            this.step();
        flush();
        const total = (Date.now() - start) / stepN;

        const ops = [];

        ops.push('Perception')


        for (let i = 0; i < this.layers.length; ++i)
            ops.push(`FC Layer${i + 1}`);
        ops.push('Stochastic Update');
        let perOpTotal = 0.0;
        const perOp = [];
        for (const op of ops) {
            const start = Date.now();
            for (let i = 0; i < stepN; ++i) {
                this.step(op);
            }
            flush(this.buf[op]);
            const dt = (Date.now() - start) / stepN;
            perOpTotal += dt
            perOp.push([op, dt]);
        }
        const perOpStr = perOp.map((p) => {
            const [programName, dt] = p;
            const percent = 100.0 * dt / perOpTotal;
            return `${programName}: ${percent.toFixed(1)}%`;
        }).join('\n');
        var result = `${(total).toFixed(2)} ms/step, ${(1000.0 / total).toFixed(2)} steps/sec\n` + perOpStr + '\n\n'
        // document.getElementById('log').innerHTML = `${(total).toFixed(2)} ms/step, ${(1000.0 / total).toFixed(2)} step/sec\n` + perOpStr + '\n\n'
        alert(result);
    }

    addGraft(x, y, r, brush) {
        // the model idx is passed as the brush
        this.runLayer(this.nca_progs.paint, this.buf.NewgraftWeights, {
            u_input: this.buf.graftWeights,

            u_canvas_size: [this.gl.canvas.width, this.gl.canvas.height],

            u_view: this.camera.view,
            u_projection: this.camera.projection,
            u_world: this.camera.world,

            u_positions: this.meshAttributeTextures.positions,
            u_positionsSize: [this.meshAttributes.positions.width, this.meshAttributes.positions.height],
            u_normals: this.meshAttributeTextures.normals,
            u_normalsSize: [this.meshAttributes.normals.width, this.meshAttributes.normals.height],

            u_pos: [x, y], u_r: r, u_brush: [0.03, 0, 0, 0],
            u_residual: true,
            u_reset: false,
        });
        [this.buf.graftWeights, this.buf.NewgraftWeights] = [this.buf.NewgraftWeights, this.buf.graftWeights];

    }

    resetGraft() {
        this.runLayer(this.nca_progs.paint, this.buf.graftWeights, {
            u_reset: true, u_brush: [0, 0, 0, 0],
        });
    }

    clearCircle(x, y, r, brush, reset = false) {
        // alert(this.camera.projection)
        this.runLayer(this.nca_progs.paint, this.buf.meshState, {

            u_input: this.buf.meshNewState,

            u_canvas_size: [this.gl.canvas.width, this.gl.canvas.height],

            u_pos: [x, y], u_r: r, u_brush: [0.0, 0.0, 0.0, 0.0],

            u_view: this.camera.view,
            u_projection: this.camera.projection,
            u_world: this.camera.world,

            u_positions: this.meshAttributeTextures.positions,
            u_positionsSize: [this.meshAttributes.positions.width, this.meshAttributes.positions.height],
            u_normals: this.meshAttributeTextures.normals,
            u_normalsSize: [this.meshAttributes.normals.width, this.meshAttributes.normals.height],

            u_reset: reset,
            u_residual: false,

        });
    }

    setWeights(models) {
        const gl = this.gl;
        this.layers.forEach(layer => gl.deleteTexture(layer));
        this.layers = models.layers.map(layer => createDenseInfo(gl, layer));
    }

    runLayer(program, output, inputs) {
        const gl = this.gl;
        gl.disable(gl.DEPTH_TEST);
        inputs = inputs || {};
        const uniforms = {};
        for (const name in inputs) {
            const val = inputs[name];
            if (val._type == 'tensor') {
                setTensorUniforms(uniforms, name, val);
            } else {
                uniforms[name] = val;
            }
        }
        uniforms['u_shuffleTex'] = this.shuffleTex;
        uniforms['u_shuffleOfs'] = this.shuffleOfs;
        uniforms['u_alignment'] = this.alignment;
        setTensorUniforms(uniforms, 'u_output', output);

        twgl.bindFramebufferInfo(gl, output.fbi);
        gl.useProgram(program.program);
        twgl.setBuffersAndAttributes(gl, program, this.quad);
        twgl.setUniforms(program, uniforms);
        twgl.drawBufferInfo(gl, this.quad);
        return {programName: program.name, output}
    }

    runDense(output, input, layer, texture_idx, relu = false, seed = 0, grafting = false) {
        return this.runLayer(this.nca_progs.dense, output, {
            u_input: input,
            u_weightTex: layer.tex, u_weightCoefs: layer.coefs, u_layout: layer.layout,
            u_seed: seed, u_fuzz: this.fuzz, u_updateProbability: this.updateProbability,
            bias: layer.bias, pos_emb: layer.pos_emb, relu: relu,
            grid_size: this.gridSize, u_angle: this.rotationAngle / 180.0 * Math.PI,


            // u_graft: this.buf.graftWeights,
            u_texture_idx: texture_idx,

            u_enable_graft: grafting,
            u_graft_weights: this.buf.graftWeights,

        });
    }

    draw_mesh(time,
              render_config =
                  {
                      point_light_strength: 0.35,
                      ambient_light_strength: 1.0,
                      enable_normal_map: true,
                      enable_ao_map: true,
                      enable_roughness_map: true,
                      visMode: 0,
                  }) {

        // visMode
        // 0: Color
        // 1: Albedo
        // 2: Normal
        // 3: Height
        // 4: Roughness
        // 5: AO


        const gl = this.gl
        time *= 0.001;  // convert to seconds

        // twgl.resizeCanvasToDisplaySize(gl.canvas);
        // alert(gl.canvas.width)
        // alert(gl.canvas.clientWidth)

        gl.enable(gl.DEPTH_TEST);

        ca.recompute_view_matrix();


        const sharedUniforms = {
            u_bumpiness: this.bumpiness,

            u_view: this.camera.view,
            u_projection: this.camera.projection,
            u_world: this.camera.world,

            u_neighbors: this.meshAttributeTextures.neighborhood,
            u_neighborsSize: [this.meshAttributes.neighborhood.width, this.meshAttributes.neighborhood.height],

            u_positions: this.meshAttributeTextures.positions,
            u_positionsSize: [this.meshAttributes.positions.width, this.meshAttributes.positions.height],
            u_normals: this.meshAttributeTextures.normals,
            u_normalsSize: [this.meshAttributes.normals.width, this.meshAttributes.normals.height],


            u_numVertices: this.mesh.numVertices,
            u_maxNeighbors: this.MAX_NEIGHBORS,

            u_cameraPosition: this.camera.cameraPosition,

            u_enable_normal_map: render_config.enable_normal_map,
            u_enable_ao_map: render_config.enable_ao_map,
            u_enable_roughness_map: render_config.enable_roughness_map,

            u_point_light_strength: render_config.point_light_strength,
            u_ambient_light_strength: render_config.ambient_light_strength,

            u_visMode: render_config.visMode,


        };
        setTensorUniforms(sharedUniforms, 'u_input', this.buf.meshState);

        setTensorUniforms(sharedUniforms, 'u_graft', this.buf.graftWeights);


        gl.useProgram(this.render_progs.program);
        twgl.setUniforms(this.render_progs, sharedUniforms);
        // alert(Object.keys(bufferInfo))
        // alert(bufferInfo.numElements)
        // set the attributes for this part.
        // gl.bindVertexArray(vao);
        // twgl.setBuffersAndAttributes(gl, this.render_progs, this.quad);

        // calls gl.uniform
        twgl.setBuffersAndAttributes(gl, this.render_progs, this.meshBufferInfo);


        twgl.drawBufferInfo(gl, this.meshBufferInfo);


    }
}