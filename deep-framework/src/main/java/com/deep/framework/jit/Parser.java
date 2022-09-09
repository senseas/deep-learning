package com.deep.framework.jit;

import com.deep.framework.jit.lexer.BlockLexer;
import com.deep.framework.jit.lexer.Lexer;
import com.deep.framework.jit.lexer.StringLexer;
import com.deep.framework.jit.lexer.TokenType;
import com.deep.framework.jit.statement.*;

import java.util.*;
import java.util.stream.Stream;

import static javax.lang.model.SourceVersion.isIdentifier;

public class Parser {
    public Lexer lexer = new BlockLexer(null);
    public List<Object> list = new ArrayList<>();
    public static List<TokenType> ends = new ArrayList<>();
    public static List<Statement> statements = new ArrayList();
    public static Map<TokenType, Statement> tokens = new HashMap<TokenType, Statement>() {{
        put(TokenType.PACKAGE, new PackageStatement());
        put(TokenType.IMPORT, new ImportStatement());
        put(TokenType.AT, new AnnotationStatement());
        put(TokenType.CLASS, new ClassStatement());
        put(TokenType.LBRACE, new BlockStatement());
    }};

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
            } else if (!a.isEmpty() && isIdentifier(a.concat(b))) {
                return a.concat(b);
            } else if (TokenType.startsWith(a.concat(b))) {
                return a.concat(b);
            } else {
                add(a);
                return b;
            }
        });
    }

    private void add(String a) {
        if (!a.isEmpty()) {
            if (Objects.nonNull(TokenType.getType(a))) {
                TokenType tokenType = TokenType.getType(a);
                list.add(tokenType);
            } else {
                list.add(a);
            }
        }
    }

    public static void parser(Statement parent, TokenType end, List<Object> lexers) {
        while (!lexers.isEmpty()) {
            Object o = lexers.remove(0);
            if (tokens.containsKey(o)) {
                Statement statement = tokens.get(o);
                statement.parser(parent, o, lexers);
            } else {
                ObjectStatement objectStatement = new ObjectStatement();
                objectStatement.setObject(o);
                Parser.statements.add(objectStatement);
            }
        }
    }

}
