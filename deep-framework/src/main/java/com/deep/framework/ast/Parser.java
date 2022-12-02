package com.deep.framework.ast;

import com.deep.framework.ast.declaration.ClassOrInterfaceDeclaration;
import com.deep.framework.ast.declaration.EnumDeclaration;
import com.deep.framework.ast.declaration.VariableDeclarator;
import com.deep.framework.ast.expression.*;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.literal.Literal;
import com.deep.framework.ast.statement.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import static com.deep.framework.ast.lexer.TokenType.*;
import static com.deep.framework.ast.literal.Literal.string;
import static com.deep.framework.ast.literal.Literal.strings;

public class Parser {
    List<Node> list = new ArrayList<>();

    public void add(String a) {
        if (a.isEmpty()) return;

        TokenType type = TokenType.getType(a);
        if (Objects.nonNull(type)) {
            list.add(type.getToken());
        } else if (Literal.isLiteral(a)) {
            list.add(Literal.getLiteral(a));
        } else {
            list.add(new Name(a));
        }
    }

    public void parser(String strFile) {
        Stream<String> stream = FileUtil.readFile(strFile);
        stream.reduce((a, b) -> {
            String c = a.concat(b);
            if (a.equals(string)) {
                return c;
            } else if (a.startsWith(string)) {
                if (c.endsWith(strings)) return c;
                if (!c.endsWith(string)) return c;
                add(c);
                return "";
            } else if (TokenType.isWhitespace(b)) {
                add(a);
                return "";
            } else if (TokenType.isNumber(c)) {
                return c;
            } else if (TokenType.startsWith(c)) {
                return c;
            } else if (TokenType.isIdentifier(c)) {
                return c;
            } else if (TokenType.contains(a)) {
                add(a);
                return b;
            } else if (TokenType.contains(b)) {
                add(a);
                return b;
            } else {
                return c;
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
        for (Node o : list) {
            if (o.equals(LBRACE)) {
                Node child = new BlockStatement(node, new NodeList<>());
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(RBRACE)) {
                node = node.getPrarent();
            } else if (o.equals(LPAREN)) {
                Node child = new ParametersExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(RPAREN)) {
                node = node.getPrarent();
            } else if (o.equals(LBRACK)) {
                Node child = new ArrayExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(RBRACK)) {
                node = node.getPrarent();
            } else {
                node.getChildrens().add(o);
            }
        }
    }

    public void parserStatement(Node prarent) {
        NodeList<Node> list = new NodeList<>();
        Node node = new Statement(prarent);
        for (Node o : prarent.getChildrens()) {
            if (o instanceof BlockStatement) {
                list.addAll(node, o);
                node = new Statement(prarent);
                parserStatement(o);
            } else if (o.equals(SEMI)) {
                list.add(node);
                node = new Statement(prarent);
            } else {
                parserStatement(o);
                node.getChildrens().add(o);
            }
        }
        if (!list.contains(node) && !node.getChildrens().isEmpty()) {
            list.add(node);
        }
        prarent.setChildrens(list);
    }

    public void reduce(Node node) {
        Name.parser(node);
        TypeParametersExpression.parser(node);
        ObjectCreationExpression.parser(node);
        ForStatement.parser(node);
        WhileStatement.parser(node);
        IfStatement.parser(node);
        ClassOrInterfaceDeclaration.parser(node);
        EnumDeclaration.parser(node);
        BinaryExpression.parser(node);
        AssignExpression.parser(node);
        VariableDeclarator.parser(node);
        LambdaExpression.parser(node);
        SynchronizedStatement.parser(node);
        ThrowStatement.parser(node);
        SwitchStatement.parser(node);
        TryStatement.parser(node);
        for (Node n : node.getChildrens()) {
            if (!n.getChildrens().isEmpty()) {
                reduce(n);
            }
        }
    }
}