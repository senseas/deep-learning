package com.deep.framework.ast;

import com.deep.framework.ast.declaration.ClassOrInterfaceDeclaration;
import com.deep.framework.ast.expression.*;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.statement.ForStatement;
import com.deep.framework.ast.statement.IfStatement;
import com.deep.framework.ast.statement.Statement;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import static javax.lang.model.SourceVersion.isIdentifier;

public class Parser {
    String string = "\"";
    String strings = "\\\"";
    List<Object> list = new ArrayList<>();

    public void add(String a) {
        if (a.isEmpty()) return;

        TokenType type = TokenType.getType(a);
        if (Objects.nonNull(type)) {
            list.add(type.getToken());
        } else {
            list.add(a);
        }
    }

    public void parser(String strFile) {
        Stream<String> stream = FileUtil.readFile(strFile);
        stream.reduce((a, b) -> {
            if (a.equals(string)) {
                return a.concat(b);
            } else if (a.startsWith(string)) {
                if (a.concat(b).endsWith(strings)) return a.concat(b);
                if (!a.concat(b).endsWith(string)) return a.concat(b);
                add(a.concat(b));
                return "";
            } else if (Character.isWhitespace(b.charAt(0))) {
                add(a);
                return "";
            } else if (!a.isEmpty() && isIdentifier(a.concat(b))) {
                return a.concat(b);
            } else if (TokenType.contains(a.concat(b))) {
                return a.concat(b);
            } else {
                add(a);
                return b;
            }
        });

        CompilationUnit compilationUnit = new CompilationUnit();
        parserBlockStatement(compilationUnit);
        parserStatement(compilationUnit);
        reduce(compilationUnit);
    }

    private void parserBlockStatement(Node prarent) {
        Node node = new Node(prarent);
        prarent.getChildrens().add(node);
        for (Object o : list) {
            if (o.equals(TokenType.LBRACE)) {
                Node child = new BlockStatement(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RBRACE)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LPAREN)) {
                Node child = new ParametersExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RPAREN)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LBRACK)) {
                Node child = new ArrayExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RBRACK)) {
                node = node.getPrarent();
            } else if (o instanceof String a && !a.startsWith(string)) {
                Name child = new Name((String) o);
                node.getChildrens().add(child);
            } else {
                node.getChildrens().add(o);
            }
        }
    }

    public void parserStatement(Node node) {
        List<Object> list = new NodeList<>();
        Statement statement = new Statement(node);
        for (Object o : node.getChildrens()) {
            if (o instanceof BlockStatement) {
                list.addAll(List.of(statement, o));
                statement = new Statement(node);
                parserStatement((Node) o);
            } else if (o.equals(TokenType.SEMI)) {
                list.add(statement);
                statement = new Statement(node);
            } else if (o instanceof Node) {
                parserStatement((Node) o);
                statement.getChildrens().add(o);
            } else {
                statement.getChildrens().add(o);
            }
        }
        if (list.isEmpty()) return;
        node.setChildrens(list);
    }

    public void reduce(Node node) {
        Name.parser(node);
        TypeParametersExpression.parser(node);
        ClassOrInterfaceDeclaration.parser(node);
        ForStatement.parser(node);
        IfStatement.parser(node);
        BinaryExpression.parser(node);
        AssignExpression.parser(node);
        for (Object n : node.getChildrens()) {
            if (n instanceof Node) {
                reduce((Node) n);
            }
        }
    }
}