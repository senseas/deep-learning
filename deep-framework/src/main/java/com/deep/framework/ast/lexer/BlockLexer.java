package com.deep.framework.ast.lexer;

import lombok.Data;

@Data
public class BlockLexer implements Lexer {
    private String type = "block";
    private Lexer lexer;
    public final static String start = "{", end = "}";

    public BlockLexer(Lexer lexer) {
        this.lexer = lexer;
    }

    public String getEnd() {
        return end;
    }
}