#pragma once

#include <queue>
#include "PA.h"

//http://www.icodebot.com/Playstation%204%20GPU
#define COARSE_TILE_WIDTH (8)
#define COARSE_TILE_WIDTH_POW2 (3)
#define COARSE_TILE_NUM_QUADS (COARSE_TILE_WIDTH * COARSE_TILE_WIDTH/4)

namespace gpu
{
	struct s_viewport;

	namespace rasterizer
	{
		struct TriangleScreenData
		{
			gpu::fixed8 v0[2];
			gpu::fixed8 v1[2];
			gpu::fixed8 v2[2];

			int32_t min[2];
			int32_t max[2];
		};

		struct RasterizerGradients
		{
			int64_t     c[3];

			gpu::fixed8 d_01[2];
			gpu::fixed8 d_12[2];
			gpu::fixed8 d_20[2];

			int32_t     fd_01[2];
			int32_t     fd_12[2];
			int32_t     fd_20[2];
		};

		struct VertexData
		{
			gpu::fixed v_pos[2];
			int32_t	   v_z;
			gpu::fixed v_w;

			gpu::fixed v_i;
			gpu::fixed v_j;
		};

		struct TriangleGradients
		{
			gpu::fixed i_0;
			gpu::fixed j_0;

			int32_t	   z_0;
			int32_t	   d_z[2];

			gpu::fixed d_1w[2];
			gpu::fixed d_iw[2];
			gpu::fixed d_jw[2];
			gpu::fixed d_i[2];
			gpu::fixed d_j[2];

			gpu::fixed i_w[3];
			gpu::fixed j_w[3];
			gpu::fixed inv_w[3];

			gpu::fixed area;
		};
		
		struct TileWalkDetailData
		{
			uint32_t start[2];
			uint32_t step;
		};

		struct TileWalkCoarseData
		{
			uint32_t start[2];
			uint32_t step;
			uint32_t maxstep[2];
		};
		
		struct PixelData
		{
			gpu::fixed i_persp_fixed;
			gpu::fixed j_persp_fixed;

			float i_persp;
			float j_persp;

			uint32_t posx;
			uint32_t posy;
			int32_t z;
		};

		struct PixelQuad
		{
			PixelData pixel[4];
			uint8_t flags;
			uint32_t primitiveId;
		};

		// > 0 zero area (conservative)
		// == 0 non-zero area
		uint8_t zero_area_cull(const TriangleScreenData& triangleData);

		void setup_viewport_bounds(TriangleScreenData& triangleData);
		
		void setup_rasterizer_gradients(const TriangleScreenData& trianglePos, RasterizerGradients& gradients);

		void setup_triangle_gradients(const VertexData& v0, const VertexData& v1, const VertexData& v2, TriangleGradients& gradients);
		
		void get_ij_persp(const TriangleGradients& gradients, gpu::fixed dx, gpu::fixed dy, gpu::fixed& i_out, gpu::fixed& j_out);

		void get_ij_linear(const TriangleGradients& gradients, gpu::fixed dx, gpu::fixed dy, gpu::fixed& i_out, gpu::fixed& j_out);

		void get_z(const TriangleGradients& gradients, gpu::fixed dx, gpu::fixed dy, int32_t& dz);

		void setup_pixel_data(const PA::PrimitiveData& primitiveData, uint32_t baseX, uint32_t baseY, uint8_t offset, PixelData& pixel);
		void setup_quad_data(const PA::PrimitiveData& primitiveData, uint32_t baseX, uint32_t baseY, PixelQuad& quad);
		void get_ij_persp(const PA::PrimitiveGradients& gradients, float dx, float dy, float& i_out, float& j_out);

		void setup_pixel_data(const TriangleGradients& gradients, const VertexData& v0, uint32_t baseX, uint32_t baseY, uint8_t offset, PixelData& pixel);
		void setup_quad_data(const TriangleGradients& gradients, const VertexData& v0, uint32_t baseX, uint32_t baseY, PixelQuad& quad);
		void test_quad_fine(const RasterizerGradients& data, PixelQuad& quad);

		//rasterizing
		uint8_t test_coords_fine(const gpu::rasterizer::RasterizerGradients& data, uint32_t posX, uint32_t posY);
		uint8_t test_coords_coarse(const gpu::rasterizer::RasterizerGradients& data, gpu::fixed8 x0, gpu::fixed8 y0, gpu::fixed x1, gpu::fixed y1, uint8_t& easyAccept);

		void setup_coarse_walk(TileWalkCoarseData& coarseWalkData, TriangleScreenData& triangleData);
		void get_coarse_walk_range(const TileWalkCoarseData& coarseWalkData, gpu::fixed8& x0, gpu::fixed8& y0, gpu::fixed8& x1, gpu::fixed8& y1);
		void do_coarse_walk_step(TileWalkCoarseData& coarseWalkData, bool* signalDoingCoarse);

		void setup_detail_walk(TileWalkDetailData& detailWalkData, uint32_t startX, uint32_t startY);
		void get_detail_walk_pos(const TileWalkDetailData& detailWalkData, uint32_t& posx, uint32_t& posy);
		void do_detail_walk_step(TileWalkDetailData& detailWalkData, bool* signalDoingDetail);
		extern uint32_t earlyAccepts;
		void process(std::queue<gpu::PA::PrimitiveData>& primitiveFifo, std::queue<PixelQuad>& pixelQuadFifo);
	}
}