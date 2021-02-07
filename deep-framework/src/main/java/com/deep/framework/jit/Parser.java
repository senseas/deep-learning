package com.deep.framework.jit;

import com.deep.framework.jit.lexer.BlockLexer;
import com.deep.framework.jit.lexer.Lexer;
import com.deep.framework.jit.lexer.StringLexer;
import com.deep.framework.jit.lexer.TokenType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

public class Parser {
    Lexer lexer = new BlockLexer(null);
    List<Object> list = new ArrayList<>();

    public void add(String a) {
        if (!a.isEmpty()) {
            if (Objects.nonNull(TokenType.getType(a))) {
                TokenType tokenType = TokenType.getType(a);
                list.add(tokenType);
            } else {
                list.add(a);
            }
        }
    }

    public void parser(String strFile) {
        Stream<String> stream = FileUtil.readFile(strFile);
        stream.reduce((a, b) -> {
            if (lexer.getType().equals("string")) {
                if (a.concat(b).endsWith("\\\"")) return a.concat(b);
                if (!b.equals(StringLexer.end)) return a.concat(b);
                lexer = lexer.getLexer();
                add(a.concat(b));
                return "";
            } else if (b.equals(StringLexer.start)) {
                lexer = new StringLexer(lexer);
                add(a);
                return a.concat(b);
            } else if (Character.isWhitespace(b.charAt(0))) {
                add(a);
                return "";
            } else if (Objects.nonNull(TokenType.getType(a.concat(b)))) {
                return a.concat(b);
            } else if (Objects.nonNull(TokenType.getType(a))) {
                add(a);
                return b;
            } else if (Objects.nonNull(TokenType.getType(b))) {
                add(a);
                return b;
            } else {
                return a.concat(b);
            }
        });
    }

}
