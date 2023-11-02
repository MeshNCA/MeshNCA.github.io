export class Mesh {
    constructor(wavefrontString, gl, scale = 1.0, n_subdivide=0, center=true) {
        const mesh_obj = Mesh.parseWavefrontObj(wavefrontString);

        this.setMeshData(mesh_obj);
        this.normalizeRange(scale, center);
        for (let i = 0; i < n_subdivide; i++) {
            this.subdivide();
        }
        // alert(this.vertexPositionIndices.length);
        this.calculateVertexNormals();

        const normals_len = this.vertexNormals.length / 3;
        // const uvs_len = this.vertexUVs.length / 2;

        this.meshAttributeTextures = this.getAttributeTextures(gl);
    }

    setMeshData(mesh_obj) {
        this.vertexPositions = mesh_obj.vertexPositions;
        this.vertexPositionIndices = mesh_obj.vertexPositionIndices;
        this.vertexNormals = [];
        this.vertexFaceNormals = mesh_obj.vertexFaceNormals;
        // this.vertexNormalIndices = mesh_obj.vertexPositionIndices;
        this.vertexFaceNormalIndices = mesh_obj.vertexFaceNormalIndices;
        this.numVertices = this.vertexPositions.length / 3;
    }

    subdivide() {
        const new_mesh_obj = {
            vertexFaceNormals: [],
            vertexFaceNormalIndices: [],

            vertexPositions: this.vertexPositions.slice(),
            vertexPositionIndices: [],
        }

        const edge_middle_vertices = {};
        let N = 0;

        let new_vertex_idx = this.vertexPositions.length / 3;

        for (let face_idx = 0; face_idx < this.vertexPositionIndices.length / 3; face_idx++) {
            let idx1 = this.vertexPositionIndices[face_idx * 3];
            let idx2 = this.vertexPositionIndices[face_idx * 3 + 1];
            let idx3 = this.vertexPositionIndices[face_idx * 3 + 2];

            let nidx1 = this.vertexFaceNormalIndices[face_idx * 3];
            let nidx2 = this.vertexFaceNormalIndices[face_idx * 3 + 1];
            let nidx3 = this.vertexFaceNormalIndices[face_idx * 3 + 2];

            const p1 = this.vertexPositions.slice(idx1 * 3, idx1 * 3 + 3);
            const p2 = this.vertexPositions.slice(idx2 * 3, idx2 * 3 + 3);
            const p3 = this.vertexPositions.slice(idx3 * 3, idx3 * 3 + 3);

            const n1 = this.vertexFaceNormals.slice(nidx1 * 3, nidx1 * 3 + 3);
            const n2 = this.vertexFaceNormals.slice(nidx2 * 3, nidx2 * 3 + 3);
            const n3 = this.vertexFaceNormals.slice(nidx3 * 3, nidx3 * 3 + 3);

            let get_edge_name = (i1, i2) => {
                return Math.max(i1, i2).toString() + "-" + Math.min(i1, i2).toString();
            }

            let e1 = get_edge_name(idx1, idx2);
            let e2 = get_edge_name(idx2, idx3);
            let e3 = get_edge_name(idx3, idx1);

            let create_vertex_if_not_existing = (edge_name, pos1, pos2) => {
                var mid_idx
                if (edge_name in edge_middle_vertices) {
                    mid_idx = edge_middle_vertices[edge_name];
                } else {
                    mid_idx = new_vertex_idx;
                    edge_middle_vertices[edge_name] = mid_idx;
                    new_vertex_idx += 1;
                    const pos = m4.scaleVector(m4.addVectors(pos1, pos2), 0.5);
                    // alert(new_mesh_obj.vertexPositions.length);
                    new_mesh_obj.vertexPositions.push(...pos);
                    // alert(new_mesh_obj.vertexPositions.length);
                }
                return mid_idx;
            }

            const midx1 = create_vertex_if_not_existing(e1, p1, p2);
            const midx2 = create_vertex_if_not_existing(e2, p2, p3);
            const midx3 = create_vertex_if_not_existing(e3, p3, p1);


            const nm1 = m4.normalize(m4.addVectors(n1, n2));
            const nm2 = m4.normalize(m4.addVectors(n2, n3));
            const nm3 = m4.normalize(m4.addVectors(n3, n1));


            new_mesh_obj.vertexPositionIndices.push(idx1, midx1, midx3);
            new_mesh_obj.vertexPositionIndices.push(midx1, midx2, midx3);
            new_mesh_obj.vertexPositionIndices.push(midx1, idx2, midx2);
            new_mesh_obj.vertexPositionIndices.push(midx3, midx2, idx3);


            new_mesh_obj.vertexFaceNormals.push(...n1, ...n2, ...n3, ...nm1, ...nm2, ...nm3);

            new_mesh_obj.vertexFaceNormalIndices.push(N, N + 3, N + 4);
            new_mesh_obj.vertexFaceNormalIndices.push(N + 3, N + 4, N + 5);
            new_mesh_obj.vertexFaceNormalIndices.push(N + 3, N + 1, N + 4);
            new_mesh_obj.vertexFaceNormalIndices.push(N + 5, N + 4, N + 2);
            N += 6;


        }

        // alert(new_mesh_obj.vertexPositionIndices.length);

        this.setMeshData(new_mesh_obj);

    }

    normalizeRange(scale = 1.0, center=true) {
        let obj_extents = this.getExtents();
        const obj_range = m4.subtractVectors(obj_extents.max, obj_extents.min);
        const obj_center = m4.scaleVector(m4.addVectors(obj_extents.max, obj_extents.min), 0.5);
        const max_range = Math.max(Math.max(obj_range[0], obj_range[1]), obj_range[2]);


        if (center) {
            for (let i = 0; i < this.vertexPositions.length; i += 3) {
                for (let j = 0; j < 3; ++j) {
                    if (obj_range[j] > 0.0) {
                        this.vertexPositions[i + j] = this.vertexPositions[i + j] - obj_extents.mean[j];
                    }
                }
            }
        }

        obj_extents = this.getExtents();

        for (let i = 0; i < this.vertexPositions.length; i += 3) {
            for (let j = 0; j < 3; ++j) {
                if (obj_range[j] > 0.0) {
                    this.vertexPositions[i + j] = scale * this.vertexPositions[i + j] / obj_extents.max_norm;
                }
            }
        }

    }

    static parseWavefrontObj(wavefrontString) {
        'use strict'
        var vertexInfoNameMap = {v: 'vertexPositions', vt: 'vertexUVs', vn: 'vertexFaceNormals'}

        var parsedJSON = {
            vertexNormals: [],
            vertexFaceNormals: [],
            vertexUVs: [],
            vertexPositions: [],
            vertexNormalIndices: [],
            vertexFaceNormalIndices: [],
            vertexUVIndices: [],
            vertexPositionIndices: []
        }

        var linesInWavefrontObj = wavefrontString.split('\n')

        var currentLine, currentLineTokens, vertexInfoType, i, k

        let has_normal = false;
        let has_uvs = false;
        let face_counter = 0;
        // Loop through and parse every line in our .obj file
        for (i = 0; i < linesInWavefrontObj.length; i++) {
            currentLine = linesInWavefrontObj[i]
            // Tokenize our current line
            currentLineTokens = currentLine.trim().split(/\s+/)
            // vertex position, vertex texture, or vertex normal
            vertexInfoType = vertexInfoNameMap[currentLineTokens[0]]

            if (vertexInfoType) {
                for (k = 1; k < currentLineTokens.length; k++) {
                    // if (vertexInfoType != 'vertexFaceNormals') {
                    parsedJSON[vertexInfoType].push(parseFloat(currentLineTokens[k]))
                    // }
                }
                if (vertexInfoType === 'vertexFaceNormals') {
                    has_normal = true;
                }
                if (vertexInfoType === 'vertexUVs') {
                    has_uvs = true;
                }

                continue
            }


            if (currentLineTokens[0] === 'f') {
                // Get our 4 sets of vertex, uv, and normal indices for this face
                for (k = 1; k < 4; k++) {
                    var indices = currentLineTokens[k].split('/')
                    parsedJSON.vertexPositionIndices.push(parseInt(indices[0], 10) - 1) // We zero index
                    if (indices.length > 1) {
                        if (has_uvs) {
                            parsedJSON.vertexUVIndices.push(parseInt(indices[1], 10) - 1) // our face indices
                        }
                        if (has_normal) {
                            parsedJSON.vertexFaceNormalIndices.push(parseInt(indices[2], 10) - 1) // by subtracting 1
                        }

                    }

                }

                if (!has_normal) {
                    // alert("Fucking here");
                    let len = parsedJSON.vertexPositionIndices.length;
                    let idx1 = parsedJSON.vertexPositionIndices[len - 1];
                    let idx2 = parsedJSON.vertexPositionIndices[len - 2];
                    let idx3 = parsedJSON.vertexPositionIndices[len - 3];


                    let [x1, y1, z1] = parsedJSON.vertexPositions.slice(idx1 * 3, idx1 * 3 + 3);
                    let [x2, y2, z2] = parsedJSON.vertexPositions.slice(idx2 * 3, idx2 * 3 + 3);
                    let [x3, y3, z3] = parsedJSON.vertexPositions.slice(idx3 * 3, idx3 * 3 + 3);


                    let [dx1, dy1, dz1] = [x2 - x1, y2 - y1, z2 - z1]
                    let [dx2, dy2, dz2] = [x3 - x1, y3 - y1, z3 - z1]

                    parsedJSON.vertexFaceNormals.push(-dy1 * dz2 + dz1 * dy2);
                    parsedJSON.vertexFaceNormals.push(-dz1 * dx2 + dx1 * dz2);
                    parsedJSON.vertexFaceNormals.push(-dx1 * dy2 + dy1 * dx2);




                    parsedJSON.vertexFaceNormalIndices.push(face_counter) // by subtracting 1
                    parsedJSON.vertexFaceNormalIndices.push(face_counter) // by subtracting 1
                    parsedJSON.vertexFaceNormalIndices.push(face_counter) // by subtracting 1
                }

                face_counter += 1;
            }
        }

        return parsedJSON
    }

    calculateVertexNormals() {
        const pos_len = this.vertexPositions.length / 3;


        const N = this.vertexPositionIndices.length;

        // alert(num_triangles);

        const vertex_normals = {}
        const vertex_num_faces = {}
        for (let i = 0; i < N; i++) {
            let vidx = this.vertexPositionIndices[i];
            let nidx = this.vertexFaceNormalIndices[i];

            if (vidx in vertex_normals) {
                vertex_normals[vidx][0] += this.vertexFaceNormals[nidx * 3]
                vertex_normals[vidx][1] += this.vertexFaceNormals[nidx * 3 + 1]
                vertex_normals[vidx][2] += this.vertexFaceNormals[nidx * 3 + 2]
                vertex_num_faces[vidx] += 1.0;
            } else {
                vertex_normals[vidx] = [
                    this.vertexFaceNormals[nidx * 3],
                    this.vertexFaceNormals[nidx * 3 + 1],
                    this.vertexFaceNormals[nidx * 3 + 2]]
                vertex_num_faces[vidx] = 1.0
            }
        }

        for (let i = 0; i < pos_len; ++i) {
            let [x, y, z] = vertex_normals[i];
            let d = Math.sqrt(x * x + y * y + z * z);
            this.vertexNormals.push(x / d);
            this.vertexNormals.push(y / d);
            this.vertexNormals.push(z / d);
        }

    }

    extractNeighborhoods() {
        const num_vertices = this.vertexPositions.length / 3;


        const num_triangles = this.vertexPositionIndices.length / 3;

        let mesh_max_neighbors = 0;

        const neighborhoods = {}
        for (let i = 0; i < num_triangles; i++) {
            let idx1 = this.vertexPositionIndices[3 * i];

            let idx2 = this.vertexPositionIndices[3 * i + 1];
            let idx3 = this.vertexPositionIndices[3 * i + 2];
            if (!(idx1 in neighborhoods)) {
                neighborhoods[idx1] = [idx2, idx3];
            } else {
                appendIfNotThere(neighborhoods[idx1], idx2);
                appendIfNotThere(neighborhoods[idx1], idx3);
            }

            if (!(idx2 in neighborhoods)) {
                neighborhoods[idx2] = [idx1, idx3];
            } else {
                appendIfNotThere(neighborhoods[idx2], idx1);
                appendIfNotThere(neighborhoods[idx2], idx3);
            }

            if (!(idx3 in neighborhoods)) {
                neighborhoods[idx3] = [idx1, idx2];
            } else {
                appendIfNotThere(neighborhoods[idx3], idx1);
                appendIfNotThere(neighborhoods[idx3], idx2);
            }

            mesh_max_neighbors = Math.max(neighborhoods[idx1].length, neighborhoods[idx2].length, neighborhoods[idx3].length)

        }

        this.mesh_max_neighbors = mesh_max_neighbors;
        this.MAX_NEIGHBORS = Math.ceil(this.mesh_max_neighbors / 4) * 4;
        console.assert(this.MAX_NEIGHBORS <= 20);
        console.log("Maximum number of neighbors on the mesh: ", mesh_max_neighbors);
        console.log("Number of Vertices: ", this.vertexPositions.length / 3);
        // if (num_neighbors > max_neighbors) {
        //         alert("Max neighborhood assumption is violated");
        // }

        const neighborhood_array = [];


        for (let vertex_idx = 0; vertex_idx < num_vertices; ++vertex_idx) {
            let neighbors = neighborhoods[vertex_idx];
            let num_neighbors = neighbors.length;

            for (let i = 0; i < this.MAX_NEIGHBORS; ++i) {
                let idx = i < num_neighbors ? neighbors[i] : -1;
                neighborhood_array.push(idx);
            }

        }


        return {neighborhoods, neighborhood_array};


    }


    getExtents() {
        const min = this.vertexPositions.slice(0, 3);
        const max = this.vertexPositions.slice(0, 3);
        const mean = this.vertexPositions.slice(0, 3);
        let max_norm = m4.length(min);
        for (let i = 3; i < this.vertexPositions.length; i += 3) {
            for (let j = 0; j < 3; ++j) {
                const v = this.vertexPositions[i + j];
                min[j] = Math.min(v, min[j]);
                max[j] = Math.max(v, max[j]);
                mean[j] = v + mean[j];
                max_norm = Math.max(max_norm, m4.length(this.vertexPositions.slice(i, i + 3)))
            }
        }

        let N = this.vertexPositions.length / 3
        mean[0] /= N;
        mean[1] /= N;
        mean[2] /= N;
        return {min, max, mean, max_norm};
    }

    getAttributeTextures(gl) {
        const meshAttributeTextures = {};
        let constant = -1;

        const pos_len = this.vertexPositions.length / 3;
        // const normals_len = this.vertexFaceNormals.length / 3;
        const normals_len = this.vertexNormals.length / 3;
        // const uvs_len = this.vertexUVs.length / 2;

        const neighborhood_information = this.extractNeighborhoods();


        let neighborhood_array = neighborhood_information.neighborhood_array;
        const neighborhood_len = this.numVertices * this.MAX_NEIGHBORS / 4; // We store it as RGBA
        const neighborhood_texture_width = Math.ceil(Math.sqrt(neighborhood_len));
        const neighborhood_texture_height = Math.ceil(neighborhood_len / neighborhood_texture_width);
        let padding = neighborhood_texture_width * neighborhood_texture_height - neighborhood_len;
        for (let i = 0; i < padding; i++) {
            neighborhood_array.push(constant);
            neighborhood_array.push(constant);
            neighborhood_array.push(constant);
            neighborhood_array.push(constant);
        }

        meshAttributeTextures.neighborhood = {
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
            width: neighborhood_texture_width,
            height: neighborhood_texture_height,
            src: neighborhood_array,
            minMag: gl.NEAREST,
            wrap: gl.CLAMP_TO_EDGE,
            flipY: false,
        };

        const pos_texture_width = Math.ceil(Math.sqrt(pos_len));
        const pos_texture_height = Math.ceil(pos_len / pos_texture_width);
        padding = pos_texture_width * pos_texture_height - pos_len;

        const position_array = Array.from(this.vertexPositions);
        for (let i = 0; i < padding; i++) {
            position_array.push(constant);
            position_array.push(constant);
            position_array.push(constant);
        }

        meshAttributeTextures.positions = {
            format: gl.RGB,
            internalFormat: gl.RGB32F,
            type: gl.FLOAT,
            width: pos_texture_width,
            height: pos_texture_height,
            src: position_array,
            minMag: gl.NEAREST,
            wrap: gl.CLAMP_TO_EDGE,
            flipY: false,
        };

        if (normals_len > 0) {
            const normals_texture_width = Math.ceil(Math.sqrt(normals_len));
            const normals_texture_height = Math.ceil(normals_len / normals_texture_width);
            padding = normals_texture_width * normals_texture_height - normals_len;
            // const normals_array = Array.from(this.vertexFaceNormals);
            const normals_array = Array.from(this.vertexNormals);

            for (let i = 0; i < padding; i++) {
                normals_array.push(constant);
                normals_array.push(constant);
                normals_array.push(constant);
            }


            meshAttributeTextures.normals = {
                format: gl.RGB,
                internalFormat: gl.RGB32F,
                type: gl.FLOAT,
                width: normals_texture_width,
                height: normals_texture_height,
                src: normals_array,
                min: gl.NEAREST,
                mag: gl.NEAREST,
                wrap: gl.CLAMP_TO_EDGE,
                flipY: false,
            };

        }

        // if (uvs_len > 0) {
        //     const uvs_texture_width = Math.ceil(Math.sqrt(uvs_len));
        //     const uvs_texture_height = Math.ceil(uvs_len / uvs_texture_width);
        //     padding = uvs_texture_width * uvs_texture_height - uvs_len;
        //     const uvs_array = Array.from(this.vertexUVs);
        //
        //     for (let i = 0; i < padding; i++) {
        //         uvs_array.push(constant);
        //         uvs_array.push(constant);
        //     }
        //
        //     meshAttributeTextures.uvs = {
        //         format: gl.RG,
        //         internalFormat: gl.RG32F,
        //         type: gl.FLOAT,
        //         width: uvs_texture_width,
        //         height: uvs_texture_height,
        //         src: uvs_array,
        //         min: gl.NEAREST,
        //         mag: gl.NEAREST,
        //         wrap: gl.CLAMP_TO_EDGE,
        //         flipY: false,
        //     };
        // }


        return meshAttributeTextures;
    }

}

function appendIfNotThere(list, x) {
    if (!(list.includes(x))) {
        list.push(x);
    }
}

class BoundingBox {
    constructor(min, max) {
        this.min = min;
        this.max = max;
    }

    containsPoint(point) {
        return (
            point.x >= this.min.x &&
            point.x <= this.max.x &&
            point.y >= this.min.y &&
            point.y <= this.max.y &&
            point.z >= this.min.z &&
            point.z <= this.max.z
        );
    }

    getCorner(index) {
        const x = (index & 1) ? this.max.x : this.min.x;
        const y = (index & 2) ? this.max.y : this.min.y;
        const z = (index & 4) ? this.max.z : this.min.z;
        return new Vector3(x, y, z);
    }

    intersect(ray) {
        let nearT = -Infinity;
        let farT = Infinity;

        for (let i = 0; i < 3; i++) {
            const origin = ray.origin.dim(i);
            const minVal = this.min.dim(i);
            const maxVal = this.max.dim(i);

            if (ray.direction.dim([i]) === 0) {
                if (origin < minVal || origin > maxVal) {
                    return false;
                }
            } else {
                let t1 = (minVal - origin) * ray.directionRcp.dim(i);
                let t2 = (maxVal - origin) * ray.directionRcp.dim(i);

                if (t1 > t2) {
                    [t1, t2] = [t2, t1];
                }

                nearT = Math.max(t1, nearT);
                farT = Math.min(t2, farT);

                if (!(nearT <= farT)) {
                    return false;
                }
            }
        }

        return ray.tmin <= farT && nearT <= ray.tmax;
    }


}

class Ray {
    constructor(origin, direction, tmin = 0.0001, tmax = Infinity) {
        this.origin = origin;
        this.direction = direction;
        this.directionRcp = direction.cwiseInverse();
        this.tmin = tmin;
        this.tmax = tmax;
    }

    pointAt(t) {
        return this.origin.add(this.direction.multiplyScalar(t));
    }
}

class Vector3 {
    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    dim(idx) {
        return (idx === 0) ? this.x : ((idx === 1) ? this.y : this.z);

    }

    toArray() {
        return [this.x, this.y, this.z];
    }

    // Calculates the magnitude (length) of the vector.
    norm() {
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }

    // Normalizes the vector (changes its length to 1).
    normalize() {
        const mag = this.norm();
        if (mag !== 0) {
            this.x /= mag;
            this.y /= mag;
            this.z /= mag;
        }
    }

    // Returns a new vector that is the result of adding this vector and another vector.
    add(otherVector) {
        return new Vector3(
            this.x + otherVector.x,
            this.y + otherVector.y,
            this.z + otherVector.z
        );
    }

    midPoint(otherVector) {
        return new Vector3(
            0.5 * (this.x + otherVector.x),
            0.5 * (this.y + otherVector.y),
            0.5 * (this.z + otherVector.z),
        );
    }

    // Returns a new vector that is the result of subtracting another vector from this vector.
    subtract(otherVector) {
        return new Vector3(
            this.x - otherVector.x,
            this.y - otherVector.y,
            this.z - otherVector.z
        );
    }

    // Returns a new vector that is the result of scaling this vector by a scalar value.
    scale(scalar) {
        return new Vector3(this.x * scalar, this.y * scalar, this.z * scalar);
    }

    // Calculates the dot product of this vector and another vector.
    dot(otherVector) {
        return this.x * otherVector.x + this.y * otherVector.y + this.z * otherVector.z;
    }

    // Calculates the cross product of this vector and another vector.
    cross(otherVector) {
        const x = this.y * otherVector.z - this.z * otherVector.y;
        const y = this.z * otherVector.x - this.x * otherVector.z;
        const z = this.x * otherVector.y - this.y * otherVector.x;
        return new Vector3(x, y, z);
    }

    cwiseInverse() {
        const x = (this.x === 0) ? Infinity : 1.0 / this.x;
        const y = (this.y === 0) ? Infinity : 1.0 / this.y;
        const z = (this.z === 0) ? Infinity : 1.0 / this.z;
        return new Vector3(x, y, z);
    }


    static min(v1, v2) {
        return new Vector3(Math.min(v1.x, v2.x), Math.min(v1.y, v2.y), Math.min(v1.z, v2.z));
    }

    static max(v1, v2) {
        return new Vector3(Math.max(v1.x, v2.x), Math.max(v1.y, v2.y), Math.max(v1.z, v2.z));
    }

    minComponent() {
        return Math.min(this.x, this.y, this.z);
    }

    maxComponent() {
        return Math.max(this.x, this.y, this.z);
    }


}