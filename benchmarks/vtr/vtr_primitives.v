// vtr_primitives.v - RAM primitives for VTR flow

// Single-port RAM
module single_port_ram #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 8
) (
    input clk,
    input we,
    input [DATA_WIDTH-1:0] data,
    input [ADDR_WIDTH-1:0] addr,
    output reg [DATA_WIDTH-1:0] out
);

    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    always @(posedge clk) begin
        if (we)
            mem[addr] <= data;
        out <= mem[addr];
    end

endmodule

// Dual-port RAM
module dual_port_ram #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 8
) (
    input clk,
    input we1, we2,
    input [DATA_WIDTH-1:0] data1, data2,
    input [ADDR_WIDTH-1:0] addr1, addr2,
    output reg [DATA_WIDTH-1:0] out1, out2
);

    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // Port 1: Write and Read
    always @(posedge clk) begin
        if (we1)
            mem[addr1] <= data1;
        out1 <= mem[addr1];
    end

    // Port 2: Write and Read
    always @(posedge clk) begin
        if (we2)
            mem[addr2] <= data2;
        out2 <= mem[addr2];
    end

endmodule