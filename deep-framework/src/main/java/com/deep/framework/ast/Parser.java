package com.deep.framework.ast;

import com.deep.framework.ast.lexer.BlockLexer;
import com.deep.framework.ast.lexer.Lexer;
import com.deep.framework.ast.lexer.StringLexer;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.statement.Statement;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

public class Parser {
    Lexer lexer = new BlockLexer(null);
    List<Object> list = new ArrayList<>();

    public void add(String a) {
        if (a.isEmpty()) return;

        TokenType type = TokenType.getType(a);
        if (Objects.nonNull(type)) {
            list.add(type);
        } else {
            list.add(a);
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

        Node node = new Node();
        for (Object o : list) {
            if (o.equals(TokenType.RBRACE)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LBRACE)) {
                Node child = new BlockStatement(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RPAREN)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LPAREN)) {
                Node child = new Node(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RBRACK)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LBRACK)) {
                Node child = new Node(node);
                node.getChildrens().add(child);
                node = child;
            } else {
                node.getChildrens().add(o);
            }
        }

        parserx(node);
    }

    public void parserx(Node node) {
        List<Object> list = new ArrayList<>();
        if (node instanceof BlockStatement nodes) {
            Statement a = new Statement(node);
            for (Object n : node.getChildrens()) {
                if (n instanceof BlockStatement) {
                    parserx((Node) n);
                    list.add(n);
                } else if (n.equals(TokenType.SEMI)) {
                    list.add(a);
                    a = new Statement(node);
                } else if (n instanceof Node) {
                    parserx((Node) n);
                    a.getChildrens().add(n);
                } else {
                    a.getChildrens().add(n);
                }
            }
            nodes.setChildrens(list);
        } else {
            for (Object n : node.getChildrens()) {
                if (n instanceof Node) {
                    parserx((Node) n);
                }
            }
        }
    }

}
