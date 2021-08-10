package com.deep.framework.jit;

import com.deep.framework.jit.lexer.BlockLexer;
import com.deep.framework.jit.lexer.Lexer;
import com.deep.framework.jit.lexer.StringLexer;
import com.deep.framework.jit.lexer.TokenType;
import com.deep.framework.jit.statement.BlockStatement;
import com.deep.framework.jit.statement.ImportStatement;
import com.deep.framework.jit.statement.PackageStatement;
import com.deep.framework.jit.statement.Statement;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

public class Parser {
    public Lexer lexer = new BlockLexer(null);
    public List<Object> list = new ArrayList<>();
    public static List<Statement> statementList = new ArrayList();

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
            } else if (!a.isEmpty() &&
                Character.isJavaIdentifierPart(a.charAt(a.length() - 1)) &&
                Character.isJavaIdentifierPart(b.charAt(0))) {
                return a.concat(b);
            } else if (TokenType.startsWith(a.concat(b))) {
                return a.concat(b);
            } else {
                add(a);
                return b;
            }
        });
    }

    public static void parser(Statement parent, TokenType end, List<Object> lexers) {
        while (!lexers.isEmpty()) {
            Object o = lexers.get(0);
            if (o.equals(end)) return;
            lexers.remove(0);
            new ArrayList<Statement>() {{
                add(new PackageStatement());
                add(new ImportStatement());
                add(new BlockStatement());
            }}.forEach(statement -> {
                statement.parser(parent, o, lexers);
            });
        }
    }

}
