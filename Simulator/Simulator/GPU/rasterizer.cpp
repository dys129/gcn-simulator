#include "gpu.h"

//http://www.hugi.scene.org/online/coding/hugi%2017%20-%20cotriang.htm

using namespace gpu;
using namespace rasterizer;

uint8_t gpu::rasterizer::zero_area_cull(const TriangleScreenData & triangleData)
{
	return uint8_t((triangleData.min[0] == triangleData.max[0]) || (triangleData.min[1] == triangleData.max[1]));
}

void gpu::rasterizer::setup_viewport_bounds(TriangleScreenData & triangleData)
{
	const uint32_t vport_scissor_tl = gpu::read_reg(PA_CL_VPORT_SCISSOR_TL);
	const uint32_t vport_scissor_br = gpu::read_reg(PA_CL_VPORT_SCISSOR_BR);

	const int32_t vport_min_x = vport_scissor_tl >> 16;
	const int32_t vport_min_y = vport_scissor_tl & 0xFFFF;

	const int32_t vport_max_x = vport_scissor_br >> 16;
	const int32_t vport_max_y = vport_scissor_br & 0xFFFF;

	triangleData.min[0] = (min3(triangleData.v0[0], triangleData.v1[0], triangleData.v2[0]) + 0xFF) >> 8;
	triangleData.min[1] = (min3(triangleData.v0[1], triangleData.v1[1], triangleData.v2[1]) + 0xFF) >> 8;

	triangleData.max[0] = (max3(triangleData.v0[0], triangleData.v1[0], triangleData.v2[0]) + 0xFF) >> 8;
	triangleData.max[1] = (max3(triangleData.v0[1], triangleData.v1[1], triangleData.v2[1]) + 0xFF) >> 8;

	triangleData.min[0] = clamp(triangleData.min[0], vport_min_x, vport_max_x);
	triangleData.max[0] = clamp(triangleData.max[0], vport_min_x, vport_max_x);

	triangleData.min[1] = clamp(triangleData.min[1], vport_min_y, vport_max_y);
	triangleData.max[1] = clamp(triangleData.max[1], vport_min_y, vport_max_y);
}

void gpu::rasterizer::setup_rasterizer_gradients(const TriangleScreenData & trianglePos, RasterizerGradients & gradients)
{
	gradients.d_01[0] = trianglePos.v0[0] - trianglePos.v1[0];
	gradients.d_12[0] = trianglePos.v1[0] - trianglePos.v2[0];
	gradients.d_20[0] = trianglePos.v2[0] - trianglePos.v0[0];

	gradients.d_01[1] = trianglePos.v0[1] - trianglePos.v1[1];
	gradients.d_12[1] = trianglePos.v1[1] - trianglePos.v2[1];
	gradients.d_20[1] = trianglePos.v2[1] - trianglePos.v0[1];

	gradients.fd_01[0] = gradients.d_01[0] << 8;
	gradients.fd_12[0] = gradients.d_12[0] << 8;
	gradients.fd_20[0] = gradients.d_20[0] << 8;

	gradients.fd_01[1] = gradients.d_01[1] << 8;
	gradients.fd_12[1] = gradients.d_12[1] << 8;
	gradients.fd_20[1] = gradients.d_20[1] << 8;

	gradients.c[0] = int64_t(gradients.d_01[1]) * int64_t(trianglePos.v0[0]) - int64_t(gradients.d_01[0]) * int64_t(trianglePos.v0[1]);
	gradients.c[1] = int64_t(gradients.d_12[1]) * int64_t(trianglePos.v1[0]) - int64_t(gradients.d_12[0]) * int64_t(trianglePos.v1[1]);
	gradients.c[2] = int64_t(gradients.d_20[1]) * int64_t(trianglePos.v2[0]) - int64_t(gradients.d_20[0]) * int64_t(trianglePos.v2[1]);

	//left fill rule
	if (gradients.d_01[1] < 0 || (gradients.d_01[1] == 0 && gradients.d_01[0] > 0)) gradients.c[0]++;
	if (gradients.d_12[1] < 0 || (gradients.d_12[1] == 0 && gradients.d_12[0] > 0)) gradients.c[1]++;
	if (gradients.d_20[1] < 0 || (gradients.d_20[1] == 0 && gradients.d_20[0] > 0)) gradients.c[2]++;
}

gpu::fixed Area(const gpu::fixed v0[2], const gpu::fixed v1[2], const gpu::fixed v2[2])
{
	return gpu::fixed_mul(gpu::fixed_sub(v2[0], v0[0]), gpu::fixed_sub(v1[1], v0[1])) -
		   gpu::fixed_mul(gpu::fixed_sub(v0[0], v1[0]), gpu::fixed_sub(v0[1], v2[1]));
}

void gpu::rasterizer::setup_triangle_gradients(const VertexData& v0, const VertexData& v1, const VertexData& v2, TriangleGradients& gradients)
{
	gradients.area = Area(v0.v_pos, v1.v_pos, v2.v_pos);

	gradients.i_0 = v0.v_i;
	gradients.j_0 = v0.v_j;
	gradients.z_0 = v0.v_z;

	gradients.inv_w[0] = gpu::fixed_rcp(v0.v_w);
	gradients.inv_w[1] = gpu::fixed_rcp(v1.v_w);
	gradients.inv_w[2] = gpu::fixed_rcp(v2.v_w);

	gradients.i_w[0] = gpu::fixed_mul(v0.v_i, gradients.inv_w[0]);
	gradients.i_w[1] = gpu::fixed_mul(v1.v_i, gradients.inv_w[1]);
	gradients.i_w[2] = gpu::fixed_mul(v2.v_i, gradients.inv_w[2]);

	gradients.j_w[0] = gpu::fixed_mul(v0.v_j, gradients.inv_w[0]);
	gradients.j_w[1] = gpu::fixed_mul(v1.v_j, gradients.inv_w[1]);
	gradients.j_w[2] = gpu::fixed_mul(v2.v_j, gradients.inv_w[2]);

	gpu::fixed v20[2] = { v2.v_pos[0] - v0.v_pos[0], v0.v_pos[1] - v2.v_pos[1] };
	gpu::fixed v10[2] = { v0.v_pos[0] - v1.v_pos[0], v1.v_pos[1] - v0.v_pos[1] };

	int64_t d_z0 = (gpu::fixed_generic_mul<24, int64_t>((v1.v_z - v0.v_z), gpu::fixed_generic_convert<16, 24, int64_t>(v20[1]))
					+ gpu::fixed_generic_mul<24, int64_t>((v2.v_z - v0.v_z), gpu::fixed_generic_convert<16, 24, int64_t>(v10[1])));
	int64_t d_z1 = (gpu::fixed_generic_mul<24, int64_t>((v1.v_z - v0.v_z), gpu::fixed_generic_convert<16, 24, int64_t>(v20[0]))
					+ gpu::fixed_generic_mul<24, int64_t>((v2.v_z - v0.v_z), gpu::fixed_generic_convert<16, 24, int64_t>(v10[0])));

	gradients.d_z[0] = (int32_t)gpu::fixed_generic_div<24, int64_t>(d_z0, gpu::fixed_generic_convert<16, 24, int64_t>(gradients.area));
	gradients.d_z[1] = (int32_t)gpu::fixed_generic_div<24, int64_t>(d_z1, gpu::fixed_generic_convert<16, 24, int64_t>(gradients.area));

	////setup gradients
	gradients.d_1w[0] = gpu::fixed_mul(gradients.inv_w[1] - gradients.inv_w[0], v20[1]) + gpu::fixed_mul(gradients.inv_w[2] - gradients.inv_w[0], v10[1]);
	gradients.d_1w[0] = gpu::fixed_div(gradients.d_1w[0], gradients.area);
	gradients.d_1w[1] = gpu::fixed_mul(gradients.inv_w[1] - gradients.inv_w[0], v20[0]) + gpu::fixed_mul(gradients.inv_w[2] - gradients.inv_w[0], v10[0]);
	gradients.d_1w[1] = gpu::fixed_div(gradients.d_1w[1], gradients.area);

	gradients.d_iw[0] = gpu::fixed_mul(gradients.i_w[1] - gradients.i_w[0], v20[1]) + gpu::fixed_mul(gradients.i_w[2] - gradients.i_w[0], v10[1]);
	gradients.d_iw[0] = gpu::fixed_div(gradients.d_iw[0], gradients.area);
	gradients.d_iw[1] = gpu::fixed_mul(gradients.i_w[1] - gradients.i_w[0], v20[0]) + gpu::fixed_mul(gradients.i_w[2] - gradients.i_w[0], v10[0]);
	gradients.d_iw[1] = gpu::fixed_div(gradients.d_iw[1], gradients.area);

	gradients.d_jw[0] = gpu::fixed_mul(gradients.j_w[1] - gradients.j_w[0], v20[1]) + gpu::fixed_mul(gradients.j_w[2] - gradients.j_w[0], v10[1]);
	gradients.d_jw[0] = gpu::fixed_div(gradients.d_jw[0], gradients.area);
	gradients.d_jw[1] = gpu::fixed_mul(gradients.j_w[1] - gradients.j_w[0], v20[0]) + gpu::fixed_mul(gradients.j_w[2] - gradients.j_w[0], v10[0]);
	gradients.d_jw[1] = gpu::fixed_div(gradients.d_jw[1], gradients.area);

	gradients.d_i[0] = gpu::fixed_mul(v1.v_i - v0.v_i, v20[1]) + gpu::fixed_mul(v2.v_i - v0.v_i, v10[1]);
	gradients.d_i[0] = gpu::fixed_div(gradients.d_i[0], gradients.area);
	gradients.d_i[1] = gpu::fixed_mul(v1.v_i - v0.v_i, v20[0]) + gpu::fixed_mul(v2.v_i - v0.v_i, v10[0]);
	gradients.d_i[1] = gpu::fixed_div(gradients.d_i[1], gradients.area);

	gradients.d_j[0] = gpu::fixed_mul(v1.v_j - v0.v_j, v20[1]) + gpu::fixed_mul(v2.v_j - v0.v_j, v10[1]);
	gradients.d_j[0] = gpu::fixed_div(gradients.d_j[0], gradients.area);
	gradients.d_j[1] = gpu::fixed_mul(v1.v_j - v0.v_j, v20[0]) + gpu::fixed_mul(v2.v_j - v0.v_j, v10[0]);
	gradients.d_j[1] = gpu::fixed_div(gradients.d_j[1], gradients.area);
}

void gpu::rasterizer::get_ij_persp(const TriangleGradients& gradients, gpu::fixed dx, gpu::fixed dy, gpu::fixed& i_out, gpu::fixed& j_out)
{
	gpu::fixed interp_iw = gradients.i_w[0] + gpu::fixed_mul(gradients.d_iw[0], dx) + gpu::fixed_mul(gradients.d_iw[1], dy);
	gpu::fixed interp_jw = gradients.j_w[0] + gpu::fixed_mul(gradients.d_jw[0], dx) + gpu::fixed_mul(gradients.d_jw[1], dy);
	gpu::fixed interp_1w = gradients.inv_w[0] + gpu::fixed_mul(gradients.d_1w[0], dx) + gpu::fixed_mul(gradients.d_1w[1], dy);

	gpu::fixed interp_w = gpu::fixed_rcp(interp_1w);

	i_out = gpu::fixed_mul(interp_iw, interp_w);
	j_out = gpu::fixed_mul(interp_jw, interp_w);
}

void gpu::rasterizer::get_ij_linear(const TriangleGradients& gradients, gpu::fixed dx, gpu::fixed dy, gpu::fixed& i_out, gpu::fixed& j_out)
{
	gpu::fixed interp_i = gradients.i_0 + gpu::fixed_mul(gradients.d_i[0], dx) + gpu::fixed_mul(gradients.d_i[1], dy);
	gpu::fixed interp_j = gradients.j_0 + gpu::fixed_mul(gradients.d_j[0], dx) + gpu::fixed_mul(gradients.d_j[1], dy);
	
	i_out = interp_i;
	j_out = interp_j;
}

void gpu::rasterizer::get_z(const TriangleGradients& gradients, gpu::fixed dx, gpu::fixed dy, int32_t& dz)
{
	int64_t dx_24 = gpu::fixed_generic_convert<16, 24, int64_t>(dx);
	int64_t dy_24 = gpu::fixed_generic_convert<16, 24, int64_t>(dy);

	int64_t res = (int64_t)gradients.z_0 + gpu::fixed_generic_mul<24, int64_t>(gradients.d_z[0], dx_24) + gpu::fixed_generic_mul<24, int64_t>(gradients.d_z[1], dy_24);
	dz = (int32_t)res;
}

void gpu::rasterizer::setup_pixel_data(const PA::PrimitiveData & primitiveData, uint32_t baseX, uint32_t baseY, uint8_t offset, PixelData & pixel)
{
	uint16_t offsetx, offsety;
	lin2zorder(offset, offsetx, offsety);

	pixel.posx = offsetx + baseX;
	pixel.posy = offsety + baseY;

	float cv0[2] = { float(pixel.posx), float(pixel.posy) };
	float dx = cv0[0] - primitiveData.v0.v_pos[0];
	float dy = cv0[1] - primitiveData.v0.v_pos[1];

	//gpu::rasterizer::get_z(gradients, dx, dy, pixel.z);

	gpu::rasterizer::get_ij_persp(primitiveData.gradients, dx, dy, pixel.i_persp, pixel.j_persp);
}

void gpu::rasterizer::setup_quad_data(const PA::PrimitiveData & primitiveData, uint32_t baseX, uint32_t baseY, PixelQuad & quad)
{
	quad.primitiveId = primitiveData.index;

	setup_pixel_data(primitiveData, baseX, baseY, 0, quad.pixel[0]);
	setup_pixel_data(primitiveData, baseX, baseY, 1, quad.pixel[1]);
	setup_pixel_data(primitiveData, baseX, baseY, 2, quad.pixel[2]);
	setup_pixel_data(primitiveData, baseX, baseY, 3, quad.pixel[3]);
}

void gpu::rasterizer::get_ij_persp(const PA::PrimitiveGradients & gradients, float dx, float dy, float & i_out, float & j_out)
{
	float interp_iw = gradients.i_w[0] + (gradients.d_iw[0] * dx) + (gradients.d_iw[1] * dy);
	float interp_jw = gradients.j_w[0] + (gradients.d_jw[0] * dx) + (gradients.d_jw[1] * dy);
	float interp_1w = gradients.inv_w[0] + (gradients.d_1w[0] * dx) + (gradients.d_1w[1] * dy);

	float interp_w = 1.0f / (interp_1w);

	i_out = (interp_iw * interp_w);
	j_out = (interp_jw * interp_w);
}
 
void gpu::rasterizer::setup_pixel_data(const TriangleGradients & gradients, const VertexData & v0, uint32_t baseX, uint32_t baseY, uint8_t offset, PixelData & pixel)
{
	uint16_t offsetx, offsety;
	lin2zorder(offset, offsetx, offsety);

	pixel.posx = offsetx + baseX;
	pixel.posy = offsety + baseY;

	gpu::fixed cv0[2] = { gpu::int_to_fixed(pixel.posx), gpu::int_to_fixed(pixel.posy) };
	gpu::fixed dx = cv0[0] - v0.v_pos[0];
	gpu::fixed dy = cv0[1] - v0.v_pos[1];

	gpu::rasterizer::get_z(gradients, dx, dy, pixel.z);

	gpu::rasterizer::get_ij_persp(gradients, dx, dy, pixel.i_persp_fixed, pixel.j_persp_fixed);
}

void gpu::rasterizer::setup_quad_data(const TriangleGradients & gradients, const VertexData & v0, uint32_t baseX, uint32_t baseY, PixelQuad & quad)
{
	setup_pixel_data(gradients, v0, baseX, baseY, 0, quad.pixel[0]);
	setup_pixel_data(gradients, v0, baseX, baseY, 1, quad.pixel[1]);
	setup_pixel_data(gradients, v0, baseX, baseY, 2, quad.pixel[2]);
	setup_pixel_data(gradients, v0, baseX, baseY, 3, quad.pixel[3]);
}

void gpu::rasterizer::test_quad_fine(const RasterizerGradients & data, PixelQuad & quad)
{
	quad.flags = 0;
	quad.flags |= (test_coords_fine(data, quad.pixel[0].posx, quad.pixel[0].posy) << 0);
	quad.flags |= (test_coords_fine(data, quad.pixel[1].posx, quad.pixel[1].posy) << 1);
	quad.flags |= (test_coords_fine(data, quad.pixel[2].posx, quad.pixel[2].posy) << 2);
	quad.flags |= (test_coords_fine(data, quad.pixel[3].posx, quad.pixel[3].posy) << 3);
}
 

uint8_t gpu::rasterizer::test_coords_fine(const RasterizerGradients& data, uint32_t posX, uint32_t posY)
{
	int64_t	CX1 = (int64_t)data.c[0] + int64_t(posY) * int64_t(data.fd_01[0]) - int64_t(posX) * int64_t(data.fd_01[1]);
	int64_t	CX2 = (int64_t)data.c[1] + int64_t(posY) * int64_t(data.fd_12[0]) - int64_t(posX) * int64_t(data.fd_12[1]);
	int64_t	CX3 = (int64_t)data.c[2] + int64_t(posY) * int64_t(data.fd_20[0]) - int64_t(posX) * int64_t(data.fd_20[1]);

	return (CX1 > 0 && CX2 > 0 && CX3 > 0);
}

uint8_t gpu::rasterizer::test_coords_coarse(const RasterizerGradients& data, gpu::fixed8 x0, gpu::fixed8 y0, gpu::fixed x1, gpu::fixed y1, uint8_t& easyAccept)
{
	bool a00 = ((int64_t)data.c[0] + int64_t(y0) * int64_t(data.d_01[0]) - int64_t(x0) * int64_t(data.d_01[1])) > 0;// C1 + DX12 * y0 - DY12 * x0 > 0;
	bool a10 = ((int64_t)data.c[0] + int64_t(y0) * int64_t(data.d_01[0]) - int64_t(x1) * int64_t(data.d_01[1])) > 0;// C1 + DX12 * y0 - DY12 * x1 > 0;
	bool a01 = ((int64_t)data.c[0] + int64_t(y1) * int64_t(data.d_01[0]) - int64_t(x0) * int64_t(data.d_01[1])) > 0;// C1 + DX12 * y1 - DY12 * x0 > 0;
	bool a11 = ((int64_t)data.c[0] + int64_t(y1) * int64_t(data.d_01[0]) - int64_t(x1) * int64_t(data.d_01[1])) > 0;// C1 + DX12 * y1 - DY12 * x1 > 0;
	int64_t a = (a00 << 0) | (a10 << 1) | (a01 << 2) | (a11 << 3);

	bool b00 = ((int64_t)data.c[1] + int64_t(y0) * int64_t(data.d_12[0]) - int64_t(x0) * int64_t(data.d_12[1])) > 0;// C2 + DX23 * y0 - DY23 * x0 > 0;
	bool b10 = ((int64_t)data.c[1] + int64_t(y0) * int64_t(data.d_12[0]) - int64_t(x1) * int64_t(data.d_12[1])) > 0;// C2 + DX23 * y0 - DY23 * x1 > 0;
	bool b01 = ((int64_t)data.c[1] + int64_t(y1) * int64_t(data.d_12[0]) - int64_t(x0) * int64_t(data.d_12[1])) > 0;// C2 + DX23 * y1 - DY23 * x0 > 0;
	bool b11 = ((int64_t)data.c[1] + int64_t(y1) * int64_t(data.d_12[0]) - int64_t(x1) * int64_t(data.d_12[1])) > 0;// C2 + DX23 * y1 - DY23 * x1 > 0;
	int64_t b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3);

	bool c00 = ((int64_t)data.c[2] + int64_t(y0) * int64_t(data.d_20[0]) - int64_t(x0) * int64_t(data.d_20[1])) > 0;// C3 + DX31 * y0 - DY31 * x0 > 0;
	bool c10 = ((int64_t)data.c[2] + int64_t(y0) * int64_t(data.d_20[0]) - int64_t(x1) * int64_t(data.d_20[1])) > 0;// C3 + DX31 * y0 - DY31 * x1 > 0;
	bool c01 = ((int64_t)data.c[2] + int64_t(y1) * int64_t(data.d_20[0]) - int64_t(x0) * int64_t(data.d_20[1])) > 0;// C3 + DX31 * y1 - DY31 * x0 > 0;
	bool c11 = ((int64_t)data.c[2] + int64_t(y1) * int64_t(data.d_20[0]) - int64_t(x1) * int64_t(data.d_20[1])) > 0;//C3 + DX31 * y1 - DY31 * x1 > 0;
	int64_t c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3);

	easyAccept = ((a & b & c) == 0xF);

	return (a == 0 || b == 0 || c == 0);
}

void gpu::rasterizer::setup_coarse_walk(TileWalkCoarseData& coarseWalkData, TriangleScreenData& triangleData)
{
	coarseWalkData.start[0] = (triangleData.min[0] & ~(COARSE_TILE_WIDTH-1));
	coarseWalkData.start[1] = (triangleData.min[1] & ~(COARSE_TILE_WIDTH-1));
	
	coarseWalkData.step = 0;
	coarseWalkData.maxstep[0] = (triangleData.max[0] - coarseWalkData.start[0] + 0x7) >> 3;
	coarseWalkData.maxstep[1] = (triangleData.max[1] - coarseWalkData.start[1] + 0x7) >> 3;
}

void gpu::rasterizer::get_coarse_walk_range(const TileWalkCoarseData& coarseWalkData, gpu::fixed8& x0, gpu::fixed8& y0, gpu::fixed8& x1, gpu::fixed8& y1)
{
	uint32_t x = coarseWalkData.step % coarseWalkData.maxstep[0];
	uint32_t y = coarseWalkData.step / coarseWalkData.maxstep[0];

	uint32_t posx = coarseWalkData.start[0] + (x << 3);
	uint32_t posy = coarseWalkData.start[1] + (y << 3);

	x0 = posx << 8;
	x1 = (posx + COARSE_TILE_WIDTH - 1) << 8;

	y0 = posy << 8;
	y1 = (posy + COARSE_TILE_WIDTH - 1) << 8;
}

void gpu::rasterizer::do_coarse_walk_step(TileWalkCoarseData& coarseWalkData, bool* signalDoingCoarse)
{
	++coarseWalkData.step;

	if (coarseWalkData.step > (coarseWalkData.maxstep[0] * coarseWalkData.maxstep[1]))
	{
		*signalDoingCoarse = false;
	}
}

void gpu::rasterizer::setup_detail_walk(TileWalkDetailData& detailWalkData, uint32_t startX, uint32_t startY)
{
	detailWalkData.start[0] = startX;
	detailWalkData.start[1] = startY;

	detailWalkData.step = 0;
}

void gpu::rasterizer::get_detail_walk_pos(const TileWalkDetailData& detailWalkData, uint32_t& posx, uint32_t& posy)
{
	uint16_t x, y;
	lin2zorder(detailWalkData.step, x, y);
	posx = detailWalkData.start[0] + (x << 1);
	posy = detailWalkData.start[1] + (y << 1);
}

void gpu::rasterizer::do_detail_walk_step(TileWalkDetailData& detailWalkData, bool* signalDoingDetail)
{
	++detailWalkData.step;

	if (detailWalkData.step >= COARSE_TILE_NUM_QUADS)
	{
		*signalDoingDetail = false;
	}
}

uint32_t gpu::rasterizer::earlyAccepts = 0;

void gpu::rasterizer::process(std::queue<gpu::PA::PrimitiveData>& primitiveFifo, std::queue<PixelQuad>& pixelQuadFifo)
{
	if (primitiveFifo.empty())
		return;

	gpu::PA::PrimitiveData nextPrim = primitiveFifo.front();
	primitiveFifo.pop();

	//gpuAssert(nextPrim.gradients.area > 0.0f);
	//sort
	//if (nextPrim.v0.v_pos[1] < nextPrim.v1.v_pos[1]) std::swap(nextPrim.v0, nextPrim.v1);
	//if (nextPrim.v0.v_pos[1] < nextPrim.v2.v_pos[1]) std::swap(nextPrim.v0, nextPrim.v2);
	//if (nextPrim.v1.v_pos[1] < nextPrim.v2.v_pos[1]) std::swap(nextPrim.v1, nextPrim.v2);
	//if (nextPrim.v0.v_pos[1] < nextPrim.v1.v_pos[1]) std::swap(nextPrim.v0.v_pos, nextPrim.v1.v_pos);
	//if (nextPrim.v0.v_pos[1] < nextPrim.v2.v_pos[1]) std::swap(nextPrim.v0.v_pos, nextPrim.v2.v_pos);
	//if (nextPrim.v1.v_pos[1] < nextPrim.v2.v_pos[1]) std::swap(nextPrim.v1.v_pos, nextPrim.v2.v_pos);

	gpu::rasterizer::TriangleScreenData		triangleData;

	triangleData = { gpu::fixed8(nextPrim.v0.v_pos[0] * 256.0f), gpu::fixed8(nextPrim.v0.v_pos[1] * 256.0f),
					 gpu::fixed8(nextPrim.v1.v_pos[0] * 256.0f), gpu::fixed8(nextPrim.v1.v_pos[1] * 256.0f), 
					 gpu::fixed8(nextPrim.v2.v_pos[0] * 256.0f), gpu::fixed8(nextPrim.v2.v_pos[1] * 256.0f), 
					 0, 0, 0, 0 };

	//if (triangleData.v0[1] < triangleData.v1[1]) std::swap(triangleData.v0, triangleData.v1);
	//if (triangleData.v0[1] < triangleData.v2[1]) std::swap(triangleData.v0, triangleData.v2);
	//if (triangleData.v1[1] < triangleData.v2[1]) std::swap(triangleData.v1, triangleData.v2);

	if (nextPrim.gradients.area < 0.0f)
		std::swap(triangleData.v0, triangleData.v2);

	//uint32_t ar = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
	//(Area(triangleData.v0, triangleData.v1, triangleData.v2))
	//std::swap(triangleData.v0, triangleData.v2);

	gpu::rasterizer::setup_viewport_bounds(triangleData);

	if (gpu::rasterizer::zero_area_cull(triangleData) > 0)
	{
		return;
	}

	gpu::rasterizer::RasterizerGradients	rasterizerData;
	gpu::rasterizer::TileWalkDetailData		detailWalkData;
	gpu::rasterizer::TileWalkCoarseData		coarseWalkData;
	gpu::rasterizer::PixelQuad				pixelQuadData;

	uint32_t counterX = 0;
	uint32_t counterY = 0;

	int32_t stepCoarseX = 0;
	int32_t stepCoarseY = 0;

	triangleData.min[0] &= ~(7);
	triangleData.min[1] &= ~(7);

	stepCoarseX = triangleData.min[0];
	stepCoarseY = triangleData.min[1];

	setup_coarse_walk(coarseWalkData, triangleData);

	setup_rasterizer_gradients(triangleData, rasterizerData);

	bool isProcessingPrimitive = true;
	while (isProcessingPrimitive)
	{
		gpu::fixed8 x0a, x1a, y0a, y1a;
		get_coarse_walk_range(coarseWalkData, x0a, y0a, x1a, y1a);

		uint8_t easyCoarseAccept = false;
		int64_t coarse_test = test_coords_coarse(rasterizerData, x0a, y0a, x1a, y1a, easyCoarseAccept);

		if (easyCoarseAccept)
		{
			earlyAccepts++;
			int k = 7;
		}

		bool isProcessingDetail = false;

		if (coarse_test)
		{
			isProcessingDetail = false;
		}
		else
		{
			uint32_t stepCoarseXl = x0a >> 8;
			uint32_t stepCoarseYl = y0a >> 8;

			setup_detail_walk(detailWalkData, stepCoarseXl, stepCoarseYl);

			isProcessingDetail = true;
		}

		do_coarse_walk_step(coarseWalkData, &isProcessingPrimitive);

		while (isProcessingDetail)
		{
			uint32_t write_flags = 0;

			get_detail_walk_pos(detailWalkData, counterX, counterY);

			setup_quad_data(nextPrim, counterX, counterY, pixelQuadData);

			test_quad_fine(rasterizerData, pixelQuadData);

			if (pixelQuadData.flags > 0)
			{
				pixelQuadFifo.push(pixelQuadData);
			}
 
			do_detail_walk_step(detailWalkData, &isProcessingDetail);
		}
	}
}
