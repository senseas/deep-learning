package com.deep.framework.ast;

import com.deep.framework.ast.expression.ArrayExpression;
import com.deep.framework.ast.expression.ParamExpression;
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

import static javax.lang.model.SourceVersion.isIdentifier;

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
            } else if (!a.isEmpty() && isIdentifier(a.concat(b))) {
                return a.concat(b);
            } else if (TokenType.startsWith(a.concat(b))) {
                return a.concat(b);
            } else {
                add(a);
                return b;
            }
        });

        Node node = parserBlockStatement();
        parserStatement(node);
    }

    private Node parserBlockStatement() {
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
                Node child = new ParamExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RBRACK)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LBRACK)) {
                Node child = new ArrayExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else {
                node.getChildrens().add(o);
            }
        }
        return node;
    }

    public void parserStatement(Node node) {
        List<Object> list = new ArrayList<>();
        Statement a = new Statement(node);
        for (Object n : node.getChildrens()) {
            if (n instanceof BlockStatement) {
                list.add(a);
                list.add(n);
                a = new Statement(node);
                parserStatement((Node) n);
            } else if (n.equals(TokenType.SEMI)) {
                list.add(a);
                a = new Statement(node);
            } else if (n instanceof Node) {
                parserStatement((Node) n);
                a.getChildrens().add(n);
            } else {
                a.getChildrens().add(n);
            }
        }
        node.setChildrens(list);
    }
}


/*
else if (n instanceof ParamExpression) {
        list.add(a);
        list.add(n);
        a = new Statement(node);
        parserStatement((Node) n);
        }else if (n instanceof ArrayExpression) {
        list.add(a);
        list.add(n);
        a = new Statement(node);
        parserStatement((Node) n);
        } */
